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
#include "pair_lj_cut_induced_dipole_cut.h"
#include "atom.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "comm.h"
#include "force.h"
#include "memory.h"
#include "error.h"
#include "update.h"
#include <string.h>
#include "math_const.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

PairLJCutInducedDipoleCut::PairLJCutInducedDipoleCut(LAMMPS *lmp) : Pair(lmp)
{
  single_enable = 0;
  nmax = atom-> nmax;
  simstep = 0;
  comm_forward = 3;
  comm_reverse = 3;
}

/* ---------------------------------------------------------------------- */

PairLJCutInducedDipoleCut::~PairLJCutInducedDipoleCut()
{
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);

    memory->destroy(cut_lj);
    memory->destroy(cut_ljsq);
    memory->destroy(cut_ljsq3inv);
    memory->destroy(cut_ljsq4inv);
    memory->destroy(cut_coul);
    memory->destroy(cut_coulsq);
    memory->destroy(epsilon);
    memory->destroy(sigma);
    memory->destroy(factor);
    memory->destroy(lj1);
    memory->destroy(lj2);
    memory->destroy(lj3);
    memory->destroy(lj4);
    memory->destroy(offset);
  }
}

/* ---------------------------------------------------------------------- */

void PairLJCutInducedDipoleCut::compute(int eflag, int vflag)
{
  if (eflag || vflag) ev_setup(eflag,vflag);
  else evflag = vflag_fdotr = 0;

  // self-consistently determine induced dipoles
  simstep++;
  int converged = 0;
  double wpot = 0;
  if (oscillating) {
    for (component=0; component<3; component++) {
      scf_energy = 1e10;
      converged = 0;
      switch (component) {
      case X_COMPONENT:
        field[0] = b[0];
        field[1] = 0.0;
        field[2] = 0.0;
        break;
      case Y_COMPONENT:
        field[0] = 0.0;
        field[1] = b[1];
        field[2] = 0.0;
        break;
      case Z_COMPONENT:
        field[0] = 0.0;
        field[1] = 0.0;
        field[2] = b[2];
        break;
      }
      iterstep = 0;

      while (converged == 0) {
        converged = update_dipoles();
      }
      compute_forces(eflag, vflag);
      wpot+=scf_energy;
    }
  } else {
    scf_energy = 1e10;
    component = 0;
    iterstep = 0;
    while (converged == 0) {
      converged = update_dipoles();
    }
    compute_forces(eflag, vflag);
    wpot+=scf_energy;
  }
  // if ((comm->me==0) && (simstep-1) % 1 == 0) fprintf(screen,"scf_energy:%20.15f\n",wpot);

  if (vflag_fdotr) virial_fdotr_compute();
}

/* ---------------------------------------------------------------------- */

void PairLJCutInducedDipoleCut::compute_forces(int eflag, int vflag)
{
  int i,j,ii,jj,inum,jnum,itype,jtype;
  double qtmp,xtmp,ytmp,ztmp,delx,dely,delz,evdwl,ecoul,fx,fy,fz;
  double rsq,rinv,r2inv,r6inv,r3inv,r5inv,r7inv;
  double forcecoulx,forcecouly,forcecoulz;
  double fq,pdotp,pidotr,pjdotr,pre1,pre2,pre3,pre4;
  double forcelj,factor_coul,factor_lj;
  int *ilist,*jlist,*numneigh,**firstneigh;

  evdwl = ecoul = 0.0;

  double **x = atom->x;
  double **f = atom->f;
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

  double cut_ljsqi,cut_ljsq4i,cut_ljsq6i;

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  // loop over neighbors of my atoms

  if (comm->me==0) {
    i = ilist[0];
    jlist = firstneigh[i];
    j = jlist[0];
    // include the scf energy as a fictional particle-particle energy
    if (evflag) ev_tally_xyz(i,j,nlocal,1,0.0,scf_energy,0.0,0.0,0.0,0.0,0.0,0.0);
  }

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
      factor_coul *= 1.0 / 4.0 / MathConst::MY_PI;
      if (oscillating) factor_coul/=3;
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

          //if (rsq > cut_ljsq[itype][jtype]) {
          if (1) {
            r3inv = r2inv*rinv;
            r5inv = r3inv*r2inv;
            r7inv = r5inv*r2inv;
          } else {
            r3inv = r2inv*rinv;
            r5inv = rinv*cut_ljsq4inv[itype][jtype];
            r7inv = r3inv*cut_ljsq4inv[itype][jtype];
          };

          pdotp = mu[i][0]*mu[j][0] + mu[i][1]*mu[j][1] + mu[i][2]*mu[j][2];
          pidotr = mu[i][0]*delx + mu[i][1]*dely + mu[i][2]*delz;
          pjdotr = mu[j][0]*delx + mu[j][1]*dely + mu[j][2]*delz;

          pre1 = 3.0*r5inv*pdotp - 15.0*r7inv*pidotr*pjdotr;
          pre2 = 3.0*r5inv*pjdotr;
          pre3 = 3.0*r5inv*pidotr;

          forcecoulx += pre1*delx + pre2*mu[i][0] + pre3*mu[j][0];
          forcecouly += pre1*dely + pre2*mu[i][1] + pre3*mu[j][1];
          forcecoulz += pre1*delz + pre2*mu[i][2] + pre3*mu[j][2];

        }

        // LJ interaction
        if (rsq < cut_ljsq[itype][jtype] && (!oscillating || component==X_COMPONENT) ) {
          r6inv = r2inv*r2inv*r2inv;
          forcelj = r6inv * (lj1[itype][jtype]*r6inv - lj2[itype][jtype]);
          forcelj *= factor_lj * r2inv;
        } else forcelj = 0.0;

        // total force

        fq = factor_coul*qqrd2e;
        fx = fq*forcecoulx + delx*forcelj;
        fy = fq*forcecouly + dely*forcelj;
        fz = fq*forcecoulz + delz*forcelj;

        // force & torque accumulation

        f[i][0] += fx;
        f[i][1] += fy;
        f[i][2] += fz;

        if (newton_pair || j < nlocal) {
          f[j][0] -= fx;
          f[j][1] -= fy;
          f[j][2] -= fz;
        }

        if (eflag) {
//          if (rsq < cut_coulsq[itype][jtype]) {
//            if (rsq < cut_ljsq[itype][jtype]) {
//              r3inv = cut_ljsq3inv[itype][jtype];
//              r5inv = r3inv*r2inv;
//            }
//            ecoul = r3inv*pdotp - 3.0*r5inv*pidotr*pjdotr;
//            ecoul *= factor_coul*qqrd2e;
//          } else ecoul = 0.0;

          if (rsq < cut_ljsq[itype][jtype] && (!oscillating || component==X_COMPONENT) ) {
            evdwl = r6inv*(lj3[itype][jtype]*r6inv-lj4[itype][jtype]) -
              offset[itype][jtype];
            evdwl *= factor_lj;
          } else evdwl = 0.0;
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

void PairLJCutInducedDipoleCut::allocate()
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
  memory->create(cut_ljsq3inv,n+1,n+1,"pair:cut_ljsq3inv");
  memory->create(cut_ljsq4inv,n+1,n+1,"pair:cut_ljsq4inv");
  memory->create(cut_coul,n+1,n+1,"pair:cut_coul");
  memory->create(cut_coulsq,n+1,n+1,"pair:cut_coulsq");
  memory->create(epsilon,n+1,n+1,"pair:epsilon");
  memory->create(sigma,n+1,n+1,"pair:sigma");
  memory->create(factor,n+1,n+1,"pair:factor");
  memory->create(lj1,n+1,n+1,"pair:lj1");
  memory->create(lj2,n+1,n+1,"pair:lj2");
  memory->create(lj3,n+1,n+1,"pair:lj3");
  memory->create(lj4,n+1,n+1,"pair:lj4");
  memory->create(offset,n+1,n+1,"pair:offset");
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void PairLJCutInducedDipoleCut::settings(int narg, char **arg)
{
  if (narg < 6 || narg > 7)
    error->all(FLERR,"Incorrect args in pair_style command");

  if (strcmp(update->unit_style,"electron") == 0)
    error->all(FLERR,"Cannot (yet) use 'electron' units with dipoles");

  double theta, b0;
  theta = force->numeric(FLERR,arg[0])/180.0*MathConst::MY_PI;
  b0 = force->numeric(FLERR,arg[1]);
  oscillating = force->numeric(FLERR,arg[2]);
  b[0] = b0*sin(theta)/sqrt(2);
  b[1] = b[0];
  b[2] = b0*cos(theta);
  field[0] = b[0];
  field[1] = b[1];
  field[2] = b[2];
  chi = force->numeric(FLERR,arg[3]);
  tolerance = force->numeric(FLERR,arg[4]);
  cut_lj_global = force->numeric(FLERR,arg[5]);
  if (narg == 6) cut_coul_global = cut_lj_global;
  else cut_coul_global = force->numeric(FLERR,arg[6]);

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
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void PairLJCutInducedDipoleCut::coeff(int narg, char **arg)
{
  if (narg < 4 || narg > 6)
    error->all(FLERR,"Incorrect args for pair coefficients");
  if (!allocated) allocate();

  int ilo,ihi,jlo,jhi;
  force->bounds(FLERR,arg[0],atom->ntypes,ilo,ihi);
  force->bounds(FLERR,arg[1],atom->ntypes,jlo,jhi);

  double epsilon_one = force->numeric(FLERR,arg[2]);
  double sigma_one = force->numeric(FLERR,arg[3]);

  double cut_lj_one = cut_lj_global;
  double cut_coul_one = cut_coul_global;
  if (narg >= 5) cut_coul_one = cut_lj_one = force->numeric(FLERR,arg[4]);
  if (narg == 6) cut_coul_one = force->numeric(FLERR,arg[5]);

  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    for (int j = MAX(jlo,i); j <= jhi; j++) {
      epsilon[i][j] = epsilon_one;
      sigma[i][j] = sigma_one;
      cut_lj[i][j] = cut_lj_one;
      cut_coul[i][j] = cut_coul_one;
      setflag[i][j] = 1;
      factor[i][j] = chi * (4.0 / 3.0) * MathConst::MY_PI * sigma[i][j] * sigma[i][j] * sigma[i][j] / 8.0;
      count++;
    }
  }

  if (count == 0) error->all(FLERR,"Incorrect args for pair coefficients");
}

/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

void PairLJCutInducedDipoleCut::init_style()
{
  if (!atom->mu_flag || !atom->mu_x_flag || !atom->mu_y_flag || !atom->mu_z_flag)
    error->all(FLERR,"Pair lj/cut/induced-dipole/cut requires atom attributes mu, mu_x, mu_y, mu_z");
  neighbor->request(this,instance_me);
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double PairLJCutInducedDipoleCut::init_one(int i, int j)
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
  cut_ljsq3inv[i][j] = 1.0/(cut_ljsq[i][j]*cut_lj[i][j]);
  cut_ljsq4inv[i][j] = 1.0/(cut_ljsq[i][j]*cut_ljsq[i][j]);
  cut_coulsq[i][j] = cut_coul[i][j] * cut_coul[i][j];

  lj1[i][j] = 48.0 * epsilon[i][j] * pow(sigma[i][j],12.0);
  lj2[i][j] = 24.0 * epsilon[i][j] * pow(sigma[i][j],6.0);
  lj3[i][j] = 4.0 * epsilon[i][j] * pow(sigma[i][j],12.0);
  lj4[i][j] = 4.0 * epsilon[i][j] * pow(sigma[i][j],6.0);

  if (offset_flag) {
    double ratio = sigma[i][j] / cut_lj[i][j];
    offset[i][j] = 4.0 * epsilon[i][j] * (pow(ratio,12.0) - pow(ratio,6.0));
  } else offset[i][j] = 0.0;

  cut_ljsq[j][i] = cut_ljsq[i][j];
  cut_ljsq3inv[j][i] = cut_ljsq3inv[i][j];
  cut_ljsq4inv[j][i] = cut_ljsq4inv[i][j];
  cut_coulsq[j][i] = cut_coulsq[i][j];
  lj1[j][i] = lj1[i][j];
  lj2[j][i] = lj2[i][j];
  lj3[j][i] = lj3[i][j];
  lj4[j][i] = lj4[i][j];
  offset[j][i] = offset[i][j];

  return cut;
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairLJCutInducedDipoleCut::write_restart(FILE *fp)
{
  write_restart_settings(fp);

  int i,j;
  for (i = 1; i <= atom->ntypes; i++)
    for (j = i; j <= atom->ntypes; j++) {
      fwrite(&setflag[i][j],sizeof(int),1,fp);
      if (setflag[i][j]) {
        fwrite(&epsilon[i][j],sizeof(double),1,fp);
        fwrite(&sigma[i][j],sizeof(double),1,fp);
        fwrite(&factor[i][j],sizeof(double),1,fp);
        fwrite(&cut_lj[i][j],sizeof(double),1,fp);
        fwrite(&cut_coul[i][j],sizeof(double),1,fp);
      }
    }
}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairLJCutInducedDipoleCut::read_restart(FILE *fp)
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
          fread(&factor[i][j],sizeof(double),1,fp);
          fread(&cut_lj[i][j],sizeof(double),1,fp);
          fread(&cut_coul[i][j],sizeof(double),1,fp);
        }
        MPI_Bcast(&epsilon[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&sigma[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&factor[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&cut_lj[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&cut_coul[i][j],1,MPI_DOUBLE,0,world);
      }
    }
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairLJCutInducedDipoleCut::write_restart_settings(FILE *fp)
{
  fwrite(&cut_lj_global,sizeof(double),1,fp);
  fwrite(&cut_coul_global,sizeof(double),1,fp);
  fwrite(&offset_flag,sizeof(int),1,fp);
  fwrite(&mix_flag,sizeof(int),1,fp);
}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairLJCutInducedDipoleCut::read_restart_settings(FILE *fp)
{
  if (comm->me == 0) {
    fread(&cut_lj_global,sizeof(double),1,fp);
    fread(&cut_coul_global,sizeof(double),1,fp);
    fread(&offset_flag,sizeof(int),1,fp);
    fread(&mix_flag,sizeof(int),1,fp);
  }
  MPI_Bcast(&cut_lj_global,1,MPI_DOUBLE,0,world);
  MPI_Bcast(&cut_coul_global,1,MPI_DOUBLE,0,world);
  MPI_Bcast(&offset_flag,1,MPI_INT,0,world);
  MPI_Bcast(&mix_flag,1,MPI_INT,0,world);
}

/* ---------------------------------------------------------------------- */

int PairLJCutInducedDipoleCut::update_dipoles()
{
  int converged;
  int i,j,ii,jj,inum,jnum,itype,jtype;
  double xtmp,ytmp,ztmp,delx,dely,delz,ecoul,fx,fy,fz,fjx,fjy,fjz;
  double rsq,rinv,r2inv,r6inv,r3inv,r5inv;
  double forcecoulx,forcecouly,forcecoulz,forcecouljx,forcecouljy,forcecouljz;
  double fq,pidotr,pjdotr,pre1,pre2;
  double factor_coul;
  int *ilist,*jlist,*numneigh,**firstneigh;

  double **x = atom->x;
  double **mu_local = atom->mu;
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
  if (simstep==1 && iterstep==0) {
    int m=0;
    while (m<(atom->nlocal+atom->nghost)) {
      mu[m][0] = field[0];
      mu[m][1] = field[1];
      mu[m][2] = field[2];
      mu[m][3] = sqrt(field[0]*field[0]+field[1]*field[1]+field[2]*field[2]);
      m++;
    }
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
      factor_coul = special_coul[sbmask(j)];
      j &= NEIGHMASK;

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx*delx + dely*dely + delz*delz;
      jtype = type[j];

      if (rsq < cut_coulsq[itype][jtype]) {
        r2inv = 1.0/rsq;
        rinv = sqrt(r2inv);

        forcecoulx = forcecouly = forcecoulz = 0.0;
        forcecouljx = forcecouljy = forcecouljz = 0.0;

        r3inv = r2inv*rinv/MathConst::MY_4PI;
        r5inv = r3inv*r2inv;

        pjdotr = mu[j][0]*delx + mu[j][1]*dely + mu[j][2]*delz;
        pre1 = (3.0 * r5inv * pjdotr);
        pre2 = r3inv;

        forcecoulx += pre1*delx - pre2*mu[j][0];
        forcecouly += pre1*dely - pre2*mu[j][1];
        forcecoulz += pre1*delz - pre2*mu[j][2];

        if (newton_pair || j < nlocal) {

          pidotr = mu[i][0]*delx + mu[i][1]*dely + mu[i][2]*delz;
          pre1 = 3.0*r5inv * pidotr;

          forcecouljx += pre1*delx - pre2*mu[i][0];
          forcecouljy += pre1*dely - pre2*mu[i][1];
          forcecouljz += pre1*delz - pre2*mu[i][2];
        }
      }

      fq = factor_coul*qqrd2e;

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

  if (newton_pair) comm->reverse_comm_pair(this);

  double ne = 0.0;
  double ne_local = 0.0;
  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    itype = type[i];
    double ifactor = factor[itype][itype];
    mu[i][0] = ifactor*(field[0] + mu_local[i][0]);
    mu[i][1] = ifactor*(field[1] + mu_local[i][1]);
    mu[i][2] = ifactor*(field[2] + mu_local[i][2]);
    mu[i][3] = sqrt(mu[i][0]*mu[i][0] + mu[i][1]*mu[i][1] + mu[i][2]*mu[i][2]);
    ne_local += 0.5*(mu[i][3]*mu[i][3]);
  }

  comm->forward_comm_pair(this);

  MPI_Allreduce(&ne_local, &ne, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  double criterion = fabs((ne-scf_energy)/scf_energy);
  converged = 0;
  if (criterion < tolerance) {
    converged = 1;
  }
  else {
    scf_energy = ne;
  }

  if (converged) {
    double field_energy_local = 0.0;
    m=0;
    while (m<atom->nlocal) {
      itype = type[m];
      double ifactor = factor[itype][itype];
      // field_energy_local -= factor[itype]*mu[m][component];
      field_energy_local += 0.5*ifactor*field[component]*field[component] - 0.5*mu[m][component]*field[component];
      // fprintf(screen,"dipole%10i%20.5f%20.5f%20.5f\n", m, mu[m][0],mu[m][1],mu[m][2]);
      m++;
    }
    MPI_Allreduce(&field_energy_local, &scf_energy, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    // relative interaction energy per dipole
    if (oscillating) scf_energy/=3.0;

  }

  return converged;
}

/* ---------------------------------------------------------------------- */

int PairLJCutInducedDipoleCut::pack_forward_comm(int n, int *list, double *buf,
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

void PairLJCutInducedDipoleCut::unpack_forward_comm(int n, int first, double *buf)
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

int PairLJCutInducedDipoleCut::pack_reverse_comm(int n, int first, double *buf)
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

void PairLJCutInducedDipoleCut::unpack_reverse_comm(int n, int *list, double *buf)
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

//double PairLJCutInducedDipoleCut::memory_usage()
//{
//  double bytes = 1000 * nmax * sizeof(double);
//  return bytes;
//}
