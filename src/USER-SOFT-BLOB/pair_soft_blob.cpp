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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "pair_soft_blob.h"
#include "atom.h"
#include "comm.h"
#include "force.h"
#include "neigh_list.h"
#include "memory.h"
#include "error.h"
#include "fix.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

PairSoftBlob::PairSoftBlob(LAMMPS *lmp) : Pair(lmp)
{
  writedata = 1;
  offset_flag = 1;
  for (int index=0; index < sizeof(id_temp_global); ++index)
  {
    id_temp_global[index]=0;
  }
}

/* ---------------------------------------------------------------------- */

PairSoftBlob::~PairSoftBlob()
{
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);

    memory->destroy(cut);
    memory->destroy(pair_type);
    memory->destroy(hbb);
    memory->destroy(wbb);
    memory->destroy(r0);
    memory->destroy(soft_blob1);
    memory->destroy(offset);
  }
}

/* ---------------------------------------------------------------------- */

void PairSoftBlob::compute(int eflag, int vflag)
{
  int i,j,ii,jj,inum,jnum,itype,jtype;
  double xtmp,ytmp,ztmp,delx,dely,delz,evdwl,fpair;
  double rsq,r,factor_lj;
  int *ilist,*jlist,*numneigh,**firstneigh;

  evdwl = 0.0;
  if (eflag || vflag) ev_setup(eflag,vflag);
  else evflag = vflag_fdotr = 0;

  double **x = atom->x;
  double **f = atom->f;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  double *special_lj = force->special_lj;
  int newton_pair = force->newton_pair;
  double energy;
  double kBT;

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;
  kBT = force->boltz*(*blob_temperature);

  // loop over neighbors of my atoms

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    itype = type[i];
    jlist = firstneigh[i];
    jnum = numneigh[i];

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      j &= NEIGHMASK;
      factor_lj = special_lj[sbmask(j)];

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx*delx + dely*dely + delz*delz;
      jtype = type[j];

      if (rsq < cutsq[itype][jtype]) {

        switch (pair_type[itype][jtype]) {
          case BLOB_BLOB: {
            energy = kBT*hbb[itype][jtype] * exp( -wbb[itype][jtype] * rsq );
            fpair = factor_lj * 2 * wbb[itype][jtype] * energy;
            if (eflag) {
              evdwl = energy - kBT*offset[itype][jtype];
              evdwl *= factor_lj;
            }
            break;
          }
          case BLOB_COLLOID: {
            r = sqrt(rsq);
            energy = kBT*hbb[itype][jtype] * exp( -wbb[itype][jtype]*(r-r0[itype][jtype]-0.5));
            fpair = factor_lj * wbb[itype][jtype] * energy / r;
            if (eflag) {
              evdwl = energy - kBT*offset[itype][jtype];
              evdwl *= factor_lj;
            }
            break;
          }
          default:
          case BLOB_NONE: {
            fpair = 0.0;
            evdwl = 0.0;
            break;
          }
        }

        f[i][0] += delx*fpair;
        f[i][1] += dely*fpair;
        f[i][2] += delz*fpair;
        if (newton_pair || j < nlocal) {
          f[j][0] -= delx*fpair;
          f[j][1] -= dely*fpair;
          f[j][2] -= delz*fpair;
        }

        if (evflag) ev_tally(i,j,nlocal,newton_pair,
                             evdwl,0.0,fpair,delx,dely,delz);
      }
    }
  }

  if (vflag_fdotr) virial_fdotr_compute();
}

/* ----------------------------------------------------------------------
   allocate all arrays
------------------------------------------------------------------------- */

void PairSoftBlob::allocate()
{
  allocated = 1;
  int n = atom->ntypes;

  memory->create(setflag,n+1,n+1,"pair:setflag");
  for (int i = 1; i <= n; i++)
    for (int j = i; j <= n; j++)
      setflag[i][j] = 0;

  memory->create(cutsq,n+1,n+1,"pair:cutsq");

  memory->create(cut,n+1,n+1,"pair:cut");
  memory->create(pair_type,n+1,n+1,"pair:pair_type");
  memory->create(hbb,n+1,n+1,"pair:hbb");
  memory->create(wbb,n+1,n+1,"pair:wbb");
  memory->create(r0,n+1,n+1,"pair:r0");
  memory->create(soft_blob1,n+1,n+1,"pair:soft_blob1");
  memory->create(offset,n+1,n+1,"pair:offset");
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void PairSoftBlob::settings(int narg, char **arg)
{
  if (narg != 2) error->all(FLERR,"Illegal pair_style command");

  cut_global = force->numeric(FLERR,arg[0]);
  int index=0;
  while (arg[1][index])
  {
    id_temp_global[index] = arg[1][index];
    ++index;
  }

  // reset cutoffs that have been explicitly set

  if (allocated) {
    int i,j;
    for (i = 1; i <= atom->ntypes; i++)
      for (j = i+1; j <= atom->ntypes; j++)
        if (setflag[i][j]) cut[i][j] = cut_global;
  }
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void PairSoftBlob::coeff(int narg, char **arg)
{
  if (narg < 6 || narg > 7)
    error->all(FLERR,"Incorrect args for pair coefficients");
  if (!allocated) allocate();

  int ilo,ihi,jlo,jhi;
  force->bounds(FLERR,arg[0],atom->ntypes,ilo,ihi);
  force->bounds(FLERR,arg[1],atom->ntypes,jlo,jhi);

  int pair_type_one = force->numeric(FLERR,arg[2]);
  double hbb_one = force->numeric(FLERR,arg[3]);
  double wbb_one = force->numeric(FLERR,arg[4]);
  double r0_one = force->numeric(FLERR,arg[5]);

  double cut_one = cut_global;
  if (narg == 7) cut_one = force->numeric(FLERR,arg[6]);

  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    for (int j = MAX(jlo,i); j <= jhi; j++) {
      pair_type[i][j] = pair_type_one;
      hbb[i][j] = hbb_one;
      wbb[i][j] = wbb_one;
      r0[i][j] = r0_one;
      cut[i][j] = cut_one;
      setflag[i][j] = 1;
      count++;
    }
  }

  if (count == 0) error->all(FLERR,"Incorrect args for pair coefficients");
}

/* ----------------------------------------------------------------------
   init specific to this bond style
------------------------------------------------------------------------- */

void PairSoftBlob::init_style()
{
  neighbor->request(this,instance_me);
  /** this bit accesses the current target temperature of the thermostat
      blob_temperature points to the target                              **/
  int ifix = modify->find_fix(id_temp_global);
  Fix *temperature_fix = modify->fix[ifix];
  int dim;
  blob_temperature = (double *) temperature_fix->extract("t_target", dim);
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double PairSoftBlob::init_one(int i, int j)
{
  if (setflag[i][j] == 0) error->all(FLERR,"All pair coeffs are not set");

  soft_blob1[i][j] = 2.0*hbb[i][j]*wbb[i][j];

  if (offset_flag) {
    switch (pair_type[i][j]) {
      case BLOB_BLOB: {
        offset[i][j] = hbb[i][j]*exp(-wbb[i][j]*cut[i][j]*cut[i][j]);
        break;
      }
      case BLOB_COLLOID: {
        offset[i][j] = hbb[i][j]*exp(-wbb[i][j]*(cut[i][j]-r0[i][j]));
        break;
      }
      default:
      case BLOB_NONE: {
        offset[i][j] = 0.0;
        break;
      }
    }
  } else offset[i][j] = 0.0;

  pair_type[j][i] = pair_type[i][j];
  hbb[j][i] = hbb[i][j];
  wbb[j][i] = wbb[i][j];
  r0[j][i] = r0[i][j];
  soft_blob1[j][i] = soft_blob1[i][j];
  offset[j][i] = offset[i][j];

  return cut[i][j];
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairSoftBlob::write_restart(FILE *fp)
{
  write_restart_settings(fp);

  int i,j;
  for (i = 1; i <= atom->ntypes; i++)
    for (j = i; j <= atom->ntypes; j++) {
      fwrite(&setflag[i][j],sizeof(int),1,fp);
      if (setflag[i][j]) {
        fwrite(&pair_type[i][j],sizeof(int),1,fp);
        fwrite(&hbb[i][j],sizeof(double),1,fp);
        fwrite(&wbb[i][j],sizeof(double),1,fp);
        fwrite(&r0[i][j],sizeof(double),1,fp);
        fwrite(&cut[i][j],sizeof(double),1,fp);
      }
    }
}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairSoftBlob::read_restart(FILE *fp)
{
  read_restart_settings(fp);

  allocate();

  int i,j;
  int me = comm->me;
  for (i = 1; i <= atom->ntypes; i++)
    for (j = i; j <= atom->ntypes; j++) {
      if (me == 0) fread(&setflag[i][j],sizeof(int),1,fp);
      MPI_Bcast(&setflag[i][j],1,MPI_INT,0,world);
      if (setflag[i][j]) {
        if (me == 0) {
          fread(&pair_type[i][j],sizeof(int),1,fp);
          fread(&hbb[i][j],sizeof(double),1,fp);
          fread(&wbb[i][j],sizeof(double),1,fp);
          fread(&r0[i][j],sizeof(double),1,fp);
          fread(&cut[i][j],sizeof(double),1,fp);
        }
        MPI_Bcast(&pair_type[i][j],1,MPI_INT,0,world);
        MPI_Bcast(&hbb[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&wbb[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&r0[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&cut[i][j],1,MPI_DOUBLE,0,world);
      }
    }
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairSoftBlob::write_restart_settings(FILE *fp)
{
  fwrite(&cut_global,sizeof(double),1,fp);
  fwrite(&id_temp_global,sizeof(id_temp_global)*sizeof(char),1,fp);
  fwrite(&offset_flag,sizeof(int),1,fp);
  fwrite(&mix_flag,sizeof(int),1,fp);
}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairSoftBlob::read_restart_settings(FILE *fp)
{
  if (comm->me == 0) {
    fread(&cut_global,sizeof(double),1,fp);
    fread(&id_temp_global,sizeof(id_temp_global)*sizeof(char),1,fp);
    fread(&offset_flag,sizeof(int),1,fp);
    fread(&mix_flag,sizeof(int),1,fp);
  }
  MPI_Bcast(&cut_global,1,MPI_DOUBLE,0,world);
  MPI_Bcast(&id_temp_global,sizeof(id_temp_global),MPI_CHAR,0,world);
  MPI_Bcast(&offset_flag,1,MPI_INT,0,world);
  MPI_Bcast(&mix_flag,1,MPI_INT,0,world);
}

/* ----------------------------------------------------------------------
   proc 0 writes to data file
------------------------------------------------------------------------- */

void PairSoftBlob::write_data(FILE *fp)
{
  for (int i = 1; i <= atom->ntypes; i++)
    fprintf(fp,"%d %d %g %g %g\n",i,pair_type[i][i],hbb[i][i],wbb[i][i],r0[i][i]);
}

/* ----------------------------------------------------------------------
   proc 0 writes all pairs to data file
------------------------------------------------------------------------- */

void PairSoftBlob::write_data_all(FILE *fp)
{
  for (int i = 1; i <= atom->ntypes; i++)
    for (int j = i; j <= atom->ntypes; j++)
      fprintf(fp,"%d %d %d %g %g %g %g\n",
              i,j,pair_type[i][j],hbb[i][j],wbb[i][j],r0[i][j],cut[i][j]);
}

/* ---------------------------------------------------------------------- */

double PairSoftBlob::single(int i, int j, int itype, int jtype, double rsq,
                         double factor_coul, double factor_lj,
                         double &fforce)
{
  double energy;
  double kBT;

  kBT = force->boltz*(*blob_temperature);

  switch (pair_type[itype][jtype]) {
    case BLOB_BLOB: {
      energy = kBT*hbb[itype][jtype]*exp(-wbb[itype][jtype]*rsq);
      fforce = 2*wbb[itype][jtype]*energy;
      break;
    }
    case BLOB_COLLOID: {
      double r = sqrt(rsq);
      energy = kBT*hbb[itype][jtype] * exp( -wbb[itype][jtype]*(r-r0[itype][jtype]));
      fforce = factor_lj * wbb[itype][jtype] * energy / r;
      break;
    }
    default:
    case BLOB_NONE: {
      energy = 0.0;
      break;
    }
  }
  return factor_lj * (energy - kBT*offset[itype][jtype]);
}

/* ---------------------------------------------------------------------- */

void *PairSoftBlob::extract(const char *str, int &dim)
{
  dim = 2;
  if (strcmp(str,"pair_type") == 0) return (void *) pair_type;
  if (strcmp(str,"hbb") == 0) return (void *) hbb;
  if (strcmp(str,"r0") == 0) return (void *) r0;
  if (strcmp(str,"wbb") == 0) return (void *) wbb;
  return NULL;
}
