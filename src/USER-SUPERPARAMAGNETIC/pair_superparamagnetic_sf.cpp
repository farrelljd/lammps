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

/* ----------------------------------------------------------------------
   Contributing authors: Mario Orsi (QMUL), m.orsi@qmul.ac.uk
                         Samuel Genheden (University of Southampton)
------------------------------------------------------------------------- */

#include <math.h>
#include <stdlib.h>
#include <iostream>
#include "pair_superparamagnetic_sf.h"
#include "atom.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "comm.h"
#include "force.h"
#include "memory.h"
#include "error.h"
#include "update.h"
#include "compute.h"
#include "compute_dipole_atom.h"
#include <string.h>
#include <math.h>

using namespace LAMMPS_NS;

static int warn_single = 0;

/* ---------------------------------------------------------------------- */

PairSuperparamagneticSF::PairSuperparamagneticSF(LAMMPS *lmp) : Pair(lmp)
{
  nmax = atom-> nmax;
  simstep = 0;
  comm_forward = 3;
  comm_reverse = 3;
}

/* ---------------------------------------------------------------------- */

PairSuperparamagneticSF::~PairSuperparamagneticSF()
{
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);

    memory->destroy(cut_lj);
    memory->destroy(cut_ljsq);
    memory->destroy(cut_coul);
    memory->destroy(cut_coulsq);
    memory->destroy(epsilon);
    memory->destroy(sigma);
    memory->destroy(lj1);
    memory->destroy(lj2);
    memory->destroy(lj3);
    memory->destroy(lj4);
    memory->destroy(scale);
  }
}

void PairSuperparamagneticSF::compute(int eflag, int vflag)
{
  if (eflag || vflag) ev_setup(eflag,vflag);
  else evflag = vflag_fdotr = 0;

  // self-consistently determine induced dipoles

  simstep++;
  int converged = 0;
  if (oscillating) {
    for (component=0; component<3; component++) {
      thing = 1e10;
      converged = 0;
      field[0] = 0.0;
      field[1] = 0.0;
      field[2] = 0.0;
      field[component] = b[component];
      iterstep = 0;

      while (converged == 0) {
        converged = update_dipoles();
      }
      compute_forces(eflag, vflag);
    }
  } else {
    thing = 1e10;
    component = 0;
    iterstep = 0;
    while (converged == 0) {
      converged = update_dipoles();
    }
    compute_forces(eflag, vflag);
  }


//  int converged = 0;
//  thing = 1e10;
//  component = 0;
//  while (converged == 0) {
//    converged = update_dipoles();
//  }
//  compute_forces(eflag, vflag);

  if (vflag_fdotr) virial_fdotr_compute();
}

/* ---------------------------------------------------------------------- */

void PairSuperparamagneticSF::compute_forces(int eflag, int vflag)
{
  int i,j,ii,jj,inum,jnum,itype,jtype;
  double qtmp,xtmp,ytmp,ztmp,delx,dely,delz,evdwl,ecoul,fx,fy,fz;
  double rsq,rinv,r2inv,r6inv,r3inv,r5inv;
  double forcecoulx,forcecouly,forcecoulz,crossx,crossy,crossz;
  double tixcoul,tiycoul,tizcoul,tjxcoul,tjycoul,tjzcoul;
  double fq,pdotp,pidotr,pjdotr,pre1,pre2,pre3,pre4;
  double forcelj,factor_coul,factor_lj;
  double presf,afac,bfac,pqfac,qpfac,forceljcut,forceljsf;
  double aforcecoulx,aforcecouly,aforcecoulz;
  double bforcecoulx,bforcecouly,bforcecoulz;
  double rcutlj2inv, rcutcoul2inv,rcutlj6inv;
  int *ilist,*jlist,*numneigh,**firstneigh;

  evdwl = ecoul = 0.0;

  double **x = atom->x;
  double **f = atom->f;
  double *q = atom->q;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  double *special_coul = force->special_coul;
  double *special_lj = force->special_lj;
  int newton_pair = force->newton_pair;
  double qqrd2e = force->qqrd2e;

  double **mu;
  switch (component) {
  case X_COMPONENT:
    mu = atom->mu_x;
    break;
  case Y_COMPONENT:
    mu = atom->mu_y;
    break;
  case Z_COMPONENT:
    mu = atom->mu_z;
    break;
  }

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  // loop over neighbors of my atoms

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    qtmp = q[i];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    itype = type[i];
    jlist = firstneigh[i];
    jnum = numneigh[i];

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      factor_lj = special_lj[sbmask(j)];
      factor_coul = special_coul[sbmask(j)];
      j &= NEIGHMASK;

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx*delx + dely*dely + delz*delz;
      jtype = type[j];

      if (rsq < cutsq[itype][jtype]) {
        r2inv = 1.0/rsq;
        rinv = sqrt(r2inv);

        forcecoulx = forcecouly = forcecoulz = 0.0;

        if (rsq < cut_coulsq[itype][jtype]) {

          rcutcoul2inv=1.0/cut_coulsq[itype][jtype];

          r3inv = r2inv*rinv;
          r5inv = r3inv*r2inv;

          pdotp = mu[i][0]*mu[j][0] + mu[i][1]*mu[j][1] + mu[i][2]*mu[j][2];
          pidotr = mu[i][0]*delx + mu[i][1]*dely + mu[i][2]*delz;
          pjdotr = mu[j][0]*delx + mu[j][1]*dely + mu[j][2]*delz;

          afac = 1.0 - rsq*rsq * rcutcoul2inv*rcutcoul2inv;
          pre1 = afac * ( pdotp - 3.0 * r2inv * pidotr * pjdotr );
          aforcecoulx = pre1*delx;
          aforcecouly = pre1*dely;
          aforcecoulz = pre1*delz;

          bfac = 1.0 - 4.0*rsq*sqrt(rsq*rcutcoul2inv)*rcutcoul2inv +
            3.0*rsq*rsq*rcutcoul2inv*rcutcoul2inv;
          presf = 2.0 * r2inv * pidotr * pjdotr;
          bforcecoulx = bfac * (pjdotr*mu[i][0]+pidotr*mu[j][0]-presf*delx);
          bforcecouly = bfac * (pjdotr*mu[i][1]+pidotr*mu[j][1]-presf*dely);
          bforcecoulz = bfac * (pjdotr*mu[i][2]+pidotr*mu[j][2]-presf*delz);

          forcecoulx += 3.0 * r5inv * ( aforcecoulx + bforcecoulx );
          forcecouly += 3.0 * r5inv * ( aforcecouly + bforcecouly );
          forcecoulz += 3.0 * r5inv * ( aforcecoulz + bforcecoulz );

        }

        fq = factor_coul*qqrd2e*scale[itype][jtype];
        fx = fq*forcecoulx;
        fy = fq*forcecouly;
        fz = fq*forcecoulz;

        // force accumulation

        f[i][0] += fx;
        f[i][1] += fy;
        f[i][2] += fz;

        if (newton_pair || j < nlocal) {
          f[j][0] -= fx;
          f[j][1] -= fy;
          f[j][2] -= fz;
        }

        if (eflag) {
          if (rsq < cut_coulsq[itype][jtype]) {
            ecoul = (1.0-sqrt(rsq/cut_coulsq[itype][jtype]));
            ecoul *= ecoul;
            ecoul *= qtmp * q[j] * rinv;
            if (mu[i][3] > 0.0 && mu[j][3] > 0.0)
              ecoul += bfac * (r3inv*pdotp - 3.0*r5inv*pidotr*pjdotr);
          } else ecoul = 0.0;

          evdwl = 0.0;
        }

        if (evflag) ev_tally_xyz(i,j,nlocal,newton_pair,
                                 evdwl,ecoul,fx,fy,fz,delx,dely,delz);
      }
    }
  }
}

/* ----------------------------------------------------------------------
   allocate all arrays
------------------------------------------------------------------------- */

void PairSuperparamagneticSF::allocate()
{
  allocated = 1;
  int n = atom->ntypes;

  memory->create(setflag,n+1,n+1,"pair:setflag");
  for (int i = 1; i <= n; i++)
    for (int j = i; j <= n; j++)
      setflag[i][j] = 0;

  memory->create(cutsq,n+1,n+1,"pair:cutsq");

  memory->create(cut_lj,n+1,n+1,"pair:cut_lj");
  memory->create(cut_ljsq,n+1,n+1,"pair:cut_ljsq");
  memory->create(cut_coul,n+1,n+1,"pair:cut_coul");
  memory->create(cut_coulsq,n+1,n+1,"pair:cut_coulsq");
  memory->create(epsilon,n+1,n+1,"pair:epsilon");
  memory->create(sigma,n+1,n+1,"pair:sigma");
  memory->create(lj1,n+1,n+1,"pair:lj1");
  memory->create(lj2,n+1,n+1,"pair:lj2");
  memory->create(lj3,n+1,n+1,"pair:lj3");
  memory->create(lj4,n+1,n+1,"pair:lj4");
  memory->create(scale,n+1,n+1,"pair:scale");
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void PairSuperparamagneticSF::settings(int narg, char **arg)
{

  if (narg != 7 )
    error->all(FLERR,"Incorrect args in pair_style command");

  if (strcmp(update->unit_style,"electron") == 0)
    error->all(FLERR,"Cannot (yet) use 'electron' units with dipoles");

  b[0] = force->numeric(FLERR,arg[0]);
  b[1] = force->numeric(FLERR,arg[1]);
  b[2] = force->numeric(FLERR,arg[2]);
  oscillating = force->numeric(FLERR,arg[3]);
  field[0] = b[0];
  field[1] = b[1];
  field[2] = b[2];
  chi = force->numeric(FLERR,arg[4])/24.0;
  tolerance = force->numeric(FLERR,arg[5]);
  cut_lj_global = force->numeric(FLERR,arg[6]);
  cut_coul_global = cut_lj_global;

  // reset cutoffs that have been explicitly set

  if (allocated) {
    int i,j;
    for (i = 1; i <= atom->ntypes; i++)
      for (j = i+1; j <= atom->ntypes; j++)
        if (setflag[i][j]) {
          cut_lj[i][j] = cut_lj_global;
          cut_coul[i][j] = cut_coul_global;
        }
  }

  //int icompute = modify->find_compute("induce");
  //Compute *dipole_compute = modify->compute[icompute];
  //dipole_compute->update_dipoles();

}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void PairSuperparamagneticSF::coeff(int narg, char **arg)
{
  if (narg < 4 || narg > 8)
    error->all(FLERR,"Incorrect args for pair coefficients");
  if (!allocated) allocate();

  int ilo,ihi,jlo,jhi;
  force->bounds(FLERR,arg[0],atom->ntypes,ilo,ihi);
  force->bounds(FLERR,arg[1],atom->ntypes,jlo,jhi);

  double epsilon_one = force->numeric(FLERR,arg[2]);
  double sigma_one = force->numeric(FLERR,arg[3]);

  double cut_lj_one = cut_lj_global;
  double cut_coul_one = cut_coul_global;
  double scale_one = 1.0;
  int iarg = 4;

  if ((narg > iarg) && (strcmp(arg[iarg],"scale") != 0)) {
    cut_coul_one = cut_lj_one = force->numeric(FLERR,arg[iarg]);
    ++iarg;
  }
  if ((narg > iarg) && (strcmp(arg[iarg],"scale") != 0)) {
    cut_coul_one = force->numeric(FLERR,arg[iarg]);
    ++iarg;
  }
  if (narg > iarg) {
    if (strcmp(arg[iarg],"scale") == 0) {
      scale_one = force->numeric(FLERR,arg[iarg+1]);
      iarg += 2;
    } else error->all(FLERR,"Incorrect args for pair coefficients");
  }
  if (iarg != narg)
    error->all(FLERR,"Incorrect args for pair coefficients");

  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    for (int j = MAX(jlo,i); j <= jhi; j++) {
      epsilon[i][j] = epsilon_one;
      sigma[i][j] = sigma_one;
      cut_lj[i][j] = cut_lj_one;
      cut_coul[i][j] = cut_coul_one;
      setflag[i][j] = 1;
      scale[i][j] = scale_one;
      count++;
    }
  }

  if (count == 0) error->all(FLERR,"Incorrect args for pair coefficients");
}

/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

void PairSuperparamagneticSF::init_style()
{
  if (!atom->mu_flag || !atom->mu_x_flag || !atom->mu_y_flag || !atom->mu_z_flag)
    error->all(FLERR,"Pair superparamagnetic requires atom attributes mu, mu_x, mu_y, mu_z");

  neighbor->request(this,instance_me);
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double PairSuperparamagneticSF::init_one(int i, int j)
{
  if (setflag[i][j] == 0) {
    epsilon[i][j] = mix_energy(epsilon[i][i],epsilon[j][j],
                               sigma[i][i],sigma[j][j]);
    sigma[i][j] = mix_distance(sigma[i][i],sigma[j][j]);
    cut_lj[i][j] = mix_distance(cut_lj[i][i],cut_lj[j][j]);
    cut_coul[i][j] = mix_distance(cut_coul[i][i],cut_coul[j][j]);
  }

  double cut = MAX(cut_lj[i][j],cut_coul[i][j]);
  cut_ljsq[i][j] = cut_lj[i][j] * cut_lj[i][j];
  cut_coulsq[i][j] = cut_coul[i][j] * cut_coul[i][j];

  lj1[i][j] = 48.0 * epsilon[i][j] * pow(sigma[i][j],12.0);
  lj2[i][j] = 24.0 * epsilon[i][j] * pow(sigma[i][j],6.0);
  lj3[i][j] = 4.0 * epsilon[i][j] * pow(sigma[i][j],12.0);
  lj4[i][j] = 4.0 * epsilon[i][j] * pow(sigma[i][j],6.0);

  cut_ljsq[j][i] = cut_ljsq[i][j];
  cut_coulsq[j][i] = cut_coulsq[i][j];
  lj1[j][i] = lj1[i][j];
  lj2[j][i] = lj2[i][j];
  lj3[j][i] = lj3[i][j];
  lj4[j][i] = lj4[i][j];
  scale[j][i] = scale[i][j];

  return cut;
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairSuperparamagneticSF::write_restart(FILE *fp)
{
  write_restart_settings(fp);

  int i,j;
  for (i = 1; i <= atom->ntypes; i++)
    for (j = i; j <= atom->ntypes; j++) {
      fwrite(&setflag[i][j],sizeof(int),1,fp);
      if (setflag[i][j]) {
        fwrite(&epsilon[i][j],sizeof(double),1,fp);
        fwrite(&sigma[i][j],sizeof(double),1,fp);
        fwrite(&cut_lj[i][j],sizeof(double),1,fp);
        fwrite(&cut_coul[i][j],sizeof(double),1,fp);
        fwrite(&scale[i][j],sizeof(double),1,fp);
      }
    }
}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairSuperparamagneticSF::read_restart(FILE *fp)
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
          fread(&epsilon[i][j],sizeof(double),1,fp);
          fread(&sigma[i][j],sizeof(double),1,fp);
          fread(&cut_lj[i][j],sizeof(double),1,fp);
          fread(&cut_coul[i][j],sizeof(double),1,fp);
          fread(&scale[i][j],sizeof(double),1,fp);
        }
        MPI_Bcast(&epsilon[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&sigma[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&cut_lj[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&cut_coul[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&scale[i][j],1,MPI_DOUBLE,0,world);
      }
    }
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairSuperparamagneticSF::write_restart_settings(FILE *fp)
{
  fwrite(&cut_lj_global,sizeof(double),1,fp);
  fwrite(&cut_coul_global,sizeof(double),1,fp);
  fwrite(&mix_flag,sizeof(int),1,fp);
}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairSuperparamagneticSF::read_restart_settings(FILE *fp)
{
  if (comm->me == 0) {
    fread(&cut_lj_global,sizeof(double),1,fp);
    fread(&cut_coul_global,sizeof(double),1,fp);
    fread(&mix_flag,sizeof(int),1,fp);
  }
  MPI_Bcast(&cut_lj_global,1,MPI_DOUBLE,0,world);
  MPI_Bcast(&cut_coul_global,1,MPI_DOUBLE,0,world);
  MPI_Bcast(&mix_flag,1,MPI_INT,0,world);
}

// PairSuperparamagneticSF: calculation of force is missing (to be implemented)
double PairSuperparamagneticSF::single(int i, int j, int itype, int jtype, double rsq,
				double factor_coul, double factor_lj,
				double &fforce)
{
  double r2inv,r6inv;
  double pdotp,pidotr,pjdotr,pre1,delx,dely,delz;
  double rinv, r3inv,r5inv, rcutlj2inv, rcutcoul2inv,rcutlj6inv;
  double qtmp,xtmp,ytmp,ztmp,bfac,pqfac,qpfac, ecoul, evdwl;

  double **x = atom->x;
  double *q = atom->q;
  double **mu = atom->mu;

  if (!warn_single) {
    warn_single = 1;
    if (comm->me == 0) {
      error->warning(FLERR,"Single method for lj/sf/dipole/sf does not compute forces");
    }
  }
  qtmp = q[i];
  xtmp = x[i][0];
  ytmp = x[i][1];
  ztmp = x[i][2];

  r2inv = 1.0/rsq;
  rinv = sqrt(r2inv);
  fforce = 0.0;

  if (rsq < cut_coulsq[itype][jtype]) {
    delx = xtmp - x[j][0];
    dely = ytmp - x[j][1];
    delz = ztmp - x[j][2];
    // if (qtmp != 0.0 && q[j] != 0.0) {
    //   pre1 = qtmp*q[j]*rinv*(r2inv-1.0/cut_coulsq[itype][jtype]);
    //   forcecoulx += pre1*delx;
    //   forcecouly += pre1*dely;
    //   forcecoulz += pre1*delz;
    // }
    if (mu[i][3] > 0.0 && mu[j][3] > 0.0) {
      r3inv = r2inv*rinv;
      r5inv = r3inv*r2inv;
      rcutcoul2inv=1.0/cut_coulsq[itype][jtype];
      pdotp = mu[i][0]*mu[j][0] + mu[i][1]*mu[j][1] + mu[i][2]*mu[j][2];
      pidotr = mu[i][0]*delx + mu[i][1]*dely + mu[i][2]*delz;
      pjdotr = mu[j][0]*delx + mu[j][1]*dely + mu[j][2]*delz;
      bfac = 1.0 - 4.0*rsq*sqrt(rsq)*rcutcoul2inv*sqrt(rcutcoul2inv) +
	3.0*rsq*rsq*rcutcoul2inv*rcutcoul2inv;
    }
    if (mu[i][3] > 0.0 && q[j] != 0.0) {
      r3inv = r2inv*rinv;
      r5inv = r3inv*r2inv;
      pidotr = mu[i][0]*delx + mu[i][1]*dely + mu[i][2]*delz;
      rcutcoul2inv=1.0/cut_coulsq[itype][jtype];
      pqfac = 1.0 - 3.0*rsq*rcutcoul2inv +
	2.0*rsq*sqrt(rsq)*rcutcoul2inv*sqrt(rcutcoul2inv);
    }
    if (mu[j][3] > 0.0 && qtmp != 0.0) {
      r3inv = r2inv*rinv;
      r5inv = r3inv*r2inv;
      pjdotr = mu[j][0]*delx + mu[j][1]*dely + mu[j][2]*delz;
      rcutcoul2inv=1.0/cut_coulsq[itype][jtype];
      qpfac = 1.0 - 3.0*rsq*rcutcoul2inv +
	2.0*rsq*sqrt(rsq)*rcutcoul2inv*sqrt(rcutcoul2inv);
    }
  }
  if (rsq < cut_ljsq[itype][jtype]) {
    r6inv = r2inv*r2inv*r2inv;
    rcutlj2inv = 1.0 / cut_ljsq[itype][jtype];
    rcutlj6inv = rcutlj2inv * rcutlj2inv * rcutlj2inv;
  }

  double eng = 0.0;
  if (rsq < cut_coulsq[itype][jtype]) {
    ecoul = (1.0-sqrt(rsq)/sqrt(cut_coulsq[itype][jtype]));
    ecoul *= ecoul;
    ecoul *= qtmp * q[j] * rinv;
    if (mu[i][3] > 0.0 && mu[j][3] > 0.0)
      ecoul += bfac * (r3inv*pdotp - 3.0*r5inv*pidotr*pjdotr);
    if (mu[i][3] > 0.0 && q[j] != 0.0)
      ecoul += -q[j] * r3inv * pqfac * pidotr;
    if (mu[j][3] > 0.0 && qtmp != 0.0)
      ecoul += qtmp * r3inv * qpfac * pjdotr;
    ecoul *= factor_coul*force->qqrd2e*scale[itype][jtype];
    eng += ecoul;
  }
  if (rsq < cut_ljsq[itype][jtype]) {
    evdwl = r6inv*(lj3[itype][jtype]*r6inv-lj4[itype][jtype])+
      rcutlj6inv*(6*lj3[itype][jtype]*rcutlj6inv-3*lj4[itype][jtype])*
      rsq*rcutlj2inv+
      rcutlj6inv*(-7*lj3[itype][jtype]*rcutlj6inv+4*lj4[itype][jtype]);
    eng += evdwl*factor_lj;
  }

  return eng;
}

/* ---------------------------------------------------------------------- */

void *PairSuperparamagneticSF::extract(const char *str, int &dim)
{
  dim = 2;
  if (strcmp(str,"epsilon") == 0) return (void *) epsilon;
  if (strcmp(str,"sigma") == 0) return (void *) sigma;
  if (strcmp(str,"scale") == 0) return (void *) scale;
  return NULL;
}

/* ---------------------------------------------------------------------- */

int PairSuperparamagneticSF::update_dipoles()
{
  int converged;
  int i,j,ii,jj,inum,jnum,itype,jtype;
  double qtmp,xtmp,ytmp,ztmp,delx,dely,delz,evdwl,ecoul,fx,fy,fz,fjx,fjy,fjz;
  double rsq,rinv,r2inv,r6inv,r3inv,r5inv;
  double forcecoulx,forcecouly,forcecoulz,forcecouljx,forcecouljy,forcecouljz,crossx,crossy,crossz;
  double tixcoul,tiycoul,tizcoul,tjxcoul,tjycoul,tjzcoul;
  double fq,pdotp,pidotr,pjdotr,pre1,pre2,pre3,pre4;
  double forcelj,factor_coul,factor_lj;
  double presf,afac,bfac,pqfac,qpfac,forceljcut,forceljsf;
  double aforcecoulx,aforcecouly,aforcecoulz;
  double bforcecoulx,bforcecouly,bforcecoulz;
  double rcutlj2inv, rcutcoul2inv,rcutlj6inv;
  int *ilist,*jlist,*numneigh,**firstneigh;

  double **x = atom->x;
  double **mu_local = atom->mu;
  double *q = atom->q;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  double *special_coul = force->special_coul;
  double *special_lj = force->special_lj;
  int newton_pair = force->newton_pair;
  double qqrd2e = force->qqrd2e;

  double **mu;
  switch (component) {
  case X_COMPONENT:
    //if (comm->me==0) fprintf(screen,"x component: ");
    mu = atom->mu_x;
    break;
  case Y_COMPONENT:
    //if (comm->me==0) fprintf(screen,"y component: ");
    mu = atom->mu_y;
    break;
  case Z_COMPONENT:
    //if (comm->me==0) fprintf(screen,"z component: ");
    mu = atom->mu_z;
    break;
  }

  int m=0;
  while (m<(atom->nlocal+atom->nghost)) {
    mu_local[m][0] = 0.0;
    mu_local[m][1] = 0.0;
    mu_local[m][2] = 0.0;
    mu_local[m][3] = 0.0;
    m++;
  }

  iterstep++;

  // loop over neighbors of my atoms

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

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
      factor_lj = special_lj[sbmask(j)];
      factor_coul = special_coul[sbmask(j)];
      j &= NEIGHMASK;

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx*delx + dely*dely + delz*delz;
      jtype = type[j];

      if (rsq < cutsq[itype][jtype]) {
        r2inv = 1.0/rsq;
        rinv = sqrt(r2inv);

        forcecoulx = forcecouly = forcecoulz = 0.0;
        forcecouljx = forcecouljy = forcecouljz = 0.0;

        if (rsq < cut_coulsq[itype][jtype]) {

          rcutcoul2inv=1.0/cut_coulsq[itype][jtype];

          r3inv = r2inv*rinv;
          r5inv = r3inv*r2inv;
          pjdotr = mu[j][0]*delx + mu[j][1]*dely + mu[j][2]*delz;
          pre1 = 3.0 * r5inv * pjdotr * (1-rsq*rcutcoul2inv);
          qpfac = 1.0 - 3.0*rsq*rcutcoul2inv +
            2.0*rsq*sqrt(rsq*rcutcoul2inv)*rcutcoul2inv;
          pre2 = r3inv * qpfac;

          forcecoulx += pre1*delx - pre2*mu[j][0];
          forcecouly += pre1*dely - pre2*mu[j][1];
          forcecoulz += pre1*delz - pre2*mu[j][2];

          fq = factor_coul*qqrd2e*scale[itype][jtype];
          if (newton_pair || j < nlocal) {
            r3inv = r2inv*rinv;
            r5inv = r3inv*r2inv;
            pidotr = mu[i][0]*delx + mu[i][1]*dely + mu[i][2]*delz;
            pre1 = 3.0 * r5inv * pidotr * (1-rsq*rcutcoul2inv);
            pqfac = 1.0 - 3.0*rsq*rcutcoul2inv +
              2.0*rsq*sqrt(rsq*rcutcoul2inv)*rcutcoul2inv;
            pre2 = r3inv * pqfac;

            forcecouljx += pre1*delx - pre2*mu[i][0];
            forcecouljy += pre1*dely - pre2*mu[i][1];
            forcecouljz += pre1*delz - pre2*mu[i][2];
          }
        }

        fq = factor_coul*qqrd2e*scale[itype][jtype];
        fx = fq*forcecoulx;
        fy = fq*forcecouly;
        fz = fq*forcecoulz;
        mu_local[i][0] += fx;
        mu_local[i][1] += fy;
        mu_local[i][2] += fz;
        if (newton_pair || j < nlocal) {
          fjx = fq*forcecouljx;
          fjy = fq*forcecouljy;
          fjz = fq*forcecouljz;
          mu_local[j][0] += fjx;
          mu_local[j][1] += fjy;
          mu_local[j][2] += fjz;
        }
      }
    }
  }

  if (newton_pair) comm->reverse_comm_pair(this);

  double ne = 0.0;
  double ne_local = 0.0;
  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    mu[i][0] = field[0] + chi*mu_local[i][0];
    mu[i][1] = field[1] + chi*mu_local[i][1];
    mu[i][2] = field[2] + chi*mu_local[i][2];
    mu[i][3] = sqrt(mu[i][0]*mu[i][0] + mu[i][1]*mu[i][1] + mu[i][2]*mu[i][2]);
    ne_local += 0.5*(mu[i][3]*mu[i][3]);
  }

  comm->forward_comm_pair(this);

  MPI_Allreduce(&ne_local, &ne, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  double criterion = fabs((ne-thing)/thing);
  converged = 0;
  //if (comm->me==0) fprintf(screen,"iteration, criterion, converged:%5i%15.10f%10i\n",iterstep,criterion,criterion<tolerance);
  if (criterion < tolerance) {
    converged = 1;
  }
  else {
    thing = ne;
  }
  return converged;
}

/* ---------------------------------------------------------------------- */

int PairSuperparamagneticSF::pack_forward_comm(int n, int *list, double *buf,
                               int pbc_flag, int *pbc)
{
  int i,j,m;
  double **mu;
  switch (component) {
  case X_COMPONENT:
    mu = atom->mu_x;
    break;
  case Y_COMPONENT:
    mu = atom->mu_y;
    break;
  case Z_COMPONENT:
    mu = atom->mu_z;
    break;
  }

  m = 0;
  for (i = 0; i < n; i++) {
    j = list[i];
    buf[m++] = mu[j][0];
    buf[m++] = mu[j][1];
    buf[m++] = mu[j][2];
  }
  return m;
}

/* ---------------------------------------------------------------------- */

void PairSuperparamagneticSF::unpack_forward_comm(int n, int first, double *buf)
{
  int i,m,last;
  double **mu;
  switch (component) {
  case X_COMPONENT:
    mu = atom->mu_x;
    break;
  case Y_COMPONENT:
    mu = atom->mu_y;
    break;
  case Z_COMPONENT:
    mu = atom->mu_z;
    break;
  }

  m = 0;
  last = first + n;
  for (i = first; i < last; i++) {
    mu[i][0] = buf[m++];
    mu[i][1] = buf[m++];
    mu[i][2] = buf[m++];
  }
}

/* ---------------------------------------------------------------------- */

int PairSuperparamagneticSF::pack_reverse_comm(int n, int first, double *buf)
{
  int i,m,last;
  double **mu_local = atom->mu;

  m = 0;
  last = first + n;
  for (i = first; i < last; i++) {
    buf[m++] = mu_local[i][0];
    buf[m++] = mu_local[i][1];
    buf[m++] = mu_local[i][2];
  }
  return m;
}

/* ---------------------------------------------------------------------- */

void PairSuperparamagneticSF::unpack_reverse_comm(int n, int *list, double *buf)
{
  int i,j,m;
  double **mu_local = atom->mu;
  int nlocal = atom->nlocal;

  m = 0;
  for (i = 0; i < n; i++) {
    j = list[i];
    mu_local[j][0] += buf[m++];
    mu_local[j][1] += buf[m++];
    mu_local[j][2] += buf[m++];
  }
}

/* ----------------------------------------------------------------------
   memory usage of local atom-based arrays
------------------------------------------------------------------------- */

//double PairSuperparamagneticSF::memory_usage()
//{
//  double bytes = 1000 * nmax * sizeof(double);
//  return bytes;
//}
