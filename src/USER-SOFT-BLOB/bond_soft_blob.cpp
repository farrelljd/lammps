/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#include <math.h>
#include <stdlib.h>
#include "bond_soft_blob.h"
#include "atom.h"
#include "neighbor.h"
#include "domain.h"
#include "comm.h"
#include "force.h"
#include "memory.h"
#include "error.h"
#include "fix.h"
#include "modify.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

BondSoftBlob::BondSoftBlob(LAMMPS *lmp) : Bond(lmp)
{
  for (int index=0; index < sizeof(id_temp_global); ++index)
  {
    id_temp_global[index]=0;
    gpset=0;
  }
}

/* ---------------------------------------------------------------------- */

BondSoftBlob::~BondSoftBlob()
{
  if (allocated && !copymode) {
    memory->destroy(setflag);
    memory->destroy(k);
    memory->destroy(r0);
    memory->destroy(gp);
    memory->destroy(tf);
  }
}

/* ---------------------------------------------------------------------- */

void BondSoftBlob::get_grafting_points()
{
  tagint *tag = atom->tag;
  int i1, i2, i1_global, i2_global;
  int **bondlist = neighbor->bondlist;
  int nbondlist = neighbor->nbondlist;

  memory->create(gp_local,atom->natoms+1,atom->natoms+1,4, "bond:gp_local");
  for (int i = 1; i <= atom->natoms; i++) {
    for (int j = 1; j <= atom->natoms; j++) {
      gp_local[i][j][0] = 0.0;
      gp_local[i][j][1] = 0.0;
      gp_local[i][j][2] = 0.0;
      gp_local[j][i][0] = 0.0;
      gp_local[j][i][1] = 0.0;
      gp_local[j][i][2] = 0.0;
    }
  }

  for (int n = 0; n < nbondlist; n++) {
    i1 = bondlist[n][0];
    i2 = bondlist[n][1];
    i1_global = tag[i1];
    i2_global = tag[i2];
    if (tf[bondlist[n][2]] == BLOB_WALL)
    {
      gp_local[i1_global][i2_global][0] = atom->x[i1][0] - atom->x[i2][0];
      gp_local[i1_global][i2_global][1] = atom->x[i1][1] - atom->x[i2][1];
      gp_local[i2_global][i1_global][0] = atom->x[i2][0] - atom->x[i1][0];
      gp_local[i2_global][i1_global][1] = atom->x[i2][1] - atom->x[i1][1];
    }
  }

  if (gpset!=1) {
    for (int i1  = 0; i1 < atom->natoms; i1++) {
      for (int j1 = 0; j1 < atom->natoms; j1++) {
        for (int k1 = 0; k1 < 3; k1++) {
          MPI_Allreduce(&gp_local[i1][j1][k1],&gp[i1][j1][k1],sizeof(gp_local[i1][j1][k1]),MPI_DOUBLE,MPI_SUM,world);
        }
      }
    }
  }

  memory->destroy(gp_local);
  gpset=1;
}

/* ---------------------------------------------------------------------- */

void BondSoftBlob::get_displacement(int i1, int i2, int type, double &delx, double &dely, double &delz)
{
  double **x = atom->x;
  tagint *tag = atom->tag;
  double z1,z2,widthx,widthy;
  z1 = x[i1][2];
  z2 = x[i2][2];
  widthx = domain->boxhi[0] - domain->boxlo[0];
  widthy = domain->boxhi[1] - domain->boxlo[1];

  switch (tf[type]) {
    case BLOB_WALL: {
      int i1g = tag[i1];
      int i2g = tag[i2];

      delx = x[i1][0] - x[i2][0] - gp[i1g][i2g][0];
      dely = x[i1][1] - x[i2][1] - gp[i1g][i2g][1];
      delz = x[i1][2] - x[i2][2] - gp[i1g][i2g][2];

      if (delx > 0 & abs(delx-widthx)<delx) {
        delx -= widthx;
      } else if (delx < 0 & delx+widthx<abs(delx)) {
        delx += widthx;
      }
      if (dely > 0 & abs(dely-widthy)<dely) {
        dely -= widthy;
      } else if (dely < 0 & dely+widthy<abs(dely)) {
        dely += widthy;
      }
      break;
    }
    default: {
      delx = x[i1][0] - x[i2][0];
      dely = x[i1][1] - x[i2][1];
      delz = x[i1][2] - x[i2][2];
      break;
    }
  }
}

/* ---------------------------------------------------------------------- */

void BondSoftBlob::compute(int eflag, int vflag)
{
  int i1,i2,n,type;
  double delx,dely,delz,ebond,fbond;
  double rsq,r,dr,rk;

  ebond = 0.0;
  if (eflag || vflag) ev_setup(eflag,vflag);
  else evflag = 0;

  double **x = atom->x;
  double **f = atom->f;
  int **bondlist = neighbor->bondlist;
  int nbondlist = neighbor->nbondlist;
  int nlocal = atom->nlocal;
  int newton_bond = force->newton_bond;
  double kBT;

  if (gpset==0) {get_grafting_points();}
  kBT = force->boltz*(*blob_temperature);

  for (n = 0; n < nbondlist; n++) {
    i1 = bondlist[n][0];
    i2 = bondlist[n][1];
    type = bondlist[n][2];

    get_displacement(i1, i2, type, delx, dely, delz);

    rsq = delx*delx + dely*dely + delz*delz;
    r = sqrt(rsq);
    dr = r - r0[type];
    rk = k[type] * dr;

    // force & energy

    if (r > 0.0) fbond = -2.0*kBT*rk/r;
    else fbond = 0.0;

    if (eflag) ebond = kBT*rk*dr;

    // apply force to each of 2 atoms

    if (newton_bond || i1 < nlocal) {
      f[i1][0] += delx*fbond;
      f[i1][1] += dely*fbond;
      f[i1][2] += delz*fbond;
    }

    if (newton_bond || i2 < nlocal) {
      f[i2][0] -= delx*fbond;
      f[i2][1] -= dely*fbond;
      f[i2][2] -= delz*fbond;
    }

    if (evflag) ev_tally(i1,i2,nlocal,newton_bond,ebond,fbond,delx,dely,delz);
  }

}

/* ---------------------------------------------------------------------- */

void BondSoftBlob::allocate()
{
  allocated = 1;
  int n = atom->nbondtypes;

  memory->create(k,n+1,"bond:k");
  memory->create(r0,n+1,"bond:r0");
  memory->create(gp,atom->natoms+1,atom->natoms+1,4, "bond:gp");
  memory->create(tf,n+1,"bond:r0");
  memory->create(setflag,n+1,"bond:setflag");

  for (int i = 1; i <= n; i++) setflag[i] = 0;
  for (int i = 1; i <= atom->natoms; i++) {
    for (int j = 1; j <= atom->natoms; j++) {
      gp[i][j][0] = 0.0;
      gp[i][j][1] = 0.0;
      gp[i][j][2] = 0.0;
      gp[j][i][0] = 0.0;
      gp[j][i][1] = 0.0;
      gp[j][i][2] = 0.0;
    }
  }
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void BondSoftBlob::settings(int narg, char **arg)
{
  if (narg != 1) error->all(FLERR,"Illegal bond_style command");

  int index=0;
  while (arg[0][index])
  {
    id_temp_global[index] = arg[0][index];
    ++index;
  }
}

/* ----------------------------------------------------------------------
   set coeffs for one or more types
------------------------------------------------------------------------- */

void BondSoftBlob::coeff(int narg, char **arg)
{
  if (narg != 4) error->all(FLERR,"Incorrect args for bond coefficients");
  if (!allocated) allocate();

  int ilo,ihi;
  force->bounds(FLERR,arg[0],atom->nbondtypes,ilo,ihi);

  int tf_one = force->numeric(FLERR,arg[1]);
  double k_one = force->numeric(FLERR,arg[2]);
  double r0_one = force->numeric(FLERR,arg[3]);

  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    tf[i] = tf_one;
    k[i] = k_one;
    r0[i] = r0_one;
    setflag[i] = 1;
    count++;
  }

  if (count == 0) error->all(FLERR,"Incorrect args for bond coefficients");
}

/* ----------------------------------------------------------------------
   init specific to this bond style
------------------------------------------------------------------------- */

void BondSoftBlob::init_style()
{
  /** this bit accesses the current target temperature of the thermostat
      blob_temperature points to the target                              **/
  int ifix = modify->find_fix(id_temp_global);
  Fix *temperature_fix = modify->fix[ifix];
  int dim;
  blob_temperature = (double *) temperature_fix->extract("t_target", dim);
}

/* ----------------------------------------------------------------------
   return an equilbrium bond length
------------------------------------------------------------------------- */

double BondSoftBlob::equilibrium_distance(int i)
{
  return r0[i];
}

/* ----------------------------------------------------------------------
   proc 0 writes out coeffs to restart file
------------------------------------------------------------------------- */

void BondSoftBlob::write_restart(FILE *fp)
{
  write_restart_settings(fp);

  for (int i = 1; i <= atom->nbondtypes; i++) {
    fwrite(&setflag[i],sizeof(int),1,fp);
    if (setflag[i]) {
      fwrite(&k[i],sizeof(double),1,fp);
      fwrite(&r0[i],sizeof(double),1,fp);
      fwrite(&tf[i],sizeof(int),1,fp);
    }
  }

  fwrite(&gpset,sizeof(int),1,fp);

  for (int i = 1; i <= atom->natoms; i++) {
    for (int j = 1; j <= atom->natoms; j++) {
      fwrite(&gp[i][j][0],sizeof(double),3,fp);
    }
  }
  //memory->create(gp,atom->natoms+1,atom->natoms+1,4, "bond:gp");

}

/* ----------------------------------------------------------------------
   proc 0 reads coeffs from restart file, bcasts them
------------------------------------------------------------------------- */

void BondSoftBlob::read_restart(FILE *fp)
{
  read_restart_settings(fp);

  allocate();

  for (int i = 1; i <= atom->nbondtypes; i++) {
    if (comm->me == 0) fread(&setflag[i],sizeof(int),1,fp);
    MPI_Bcast(&setflag[i],1,MPI_INT,0,world);
    if (setflag[i]) {
      if (comm->me == 0) {
        fread(&k[i],sizeof(double),1,fp);
        fread(&r0[i],sizeof(double),1,fp);
        fread(&tf[i],sizeof(int),1,fp);
      }
      MPI_Bcast(&k[i],1,MPI_DOUBLE,0,world);
      MPI_Bcast(&r0[i],1,MPI_DOUBLE,0,world);
      MPI_Bcast(&tf[i],1,MPI_INT,0,world);
    }
  }

  if (comm->me == 0) fread(&gpset,sizeof(int),1,fp);
  MPI_Bcast(&gpset,1,MPI_INT,0,world);

  for (int i = 1; i <= atom->natoms; i++) {
    for (int j = 1; j <= atom->natoms; j++) {
      if (comm->me == 0) fread(&gp[i][j][0],sizeof(double),3,fp);
      MPI_Bcast(gp[i][j],3,MPI_DOUBLE,0,world);
    }
  }
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
------------------------------------------------------------------------- */

void BondSoftBlob::write_restart_settings(FILE *fp)
{
  fwrite(&id_temp_global,sizeof(id_temp_global)*sizeof(char),1,fp);
}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void BondSoftBlob::read_restart_settings(FILE *fp)
{
  if (comm->me == 0) {
    fread(&id_temp_global,sizeof(id_temp_global)*sizeof(char),1,fp);
  }
  MPI_Bcast(&id_temp_global,sizeof(id_temp_global),MPI_CHAR,0,world);
}

/* ----------------------------------------------------------------------
   proc 0 writes to data file
------------------------------------------------------------------------- */

void BondSoftBlob::write_data(FILE *fp)
{
  for (int i = 1; i <= atom->nbondtypes; i++)
    fprintf(fp,"%d %g %g\n",i,k[i],r0[i]);
}

/* ---------------------------------------------------------------------- */

double BondSoftBlob::single(int type, double rsq, int i, int j,
                        double &fforce)
{
  double r = sqrt(rsq);
  double dr = r - r0[type];
  double rk = k[type] * dr;
  double kBT;

  kBT = force->boltz*(*blob_temperature);
  fforce = 0;
  if (r > 0.0) fforce = -2.0*kBT*rk/r;
  return kBT*rk*dr;
}
