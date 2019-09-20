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

#include "fix_adress.h"
#include "atom.h"
#include "comm.h"
#include "variable.h"
#include "error.h"
#include "force.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "group.h"
#include "domain.h"

using namespace LAMMPS_NS;
using namespace FixConst;

/* ---------------------------------------------------------------------- */

FixAdress::FixAdress(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg)
{
  comm_forward = 3;
  comm_reverse = 3;

  // parse args

  if (narg != 8) error->all(FLERR,"Illegal fix adress command");

  int igroup;
  igroup = group->find(arg[3]);
  if (igroup == -1) error->all(FLERR,"Could not find atomistic group ID");
  atomisticbit = group->bitmask[igroup];
  igroup = group->find(arg[4]);
  if (igroup == -1) error->all(FLERR,"Could not find coarse group ID");
  coarsebit = group->bitmask[igroup];
  allbit = atomisticbit | coarsebit;

  bondtype = force->inumeric(FLERR,arg[5]);
  dex = force->numeric(FLERR,arg[6]);
  dhy = force->numeric(FLERR,arg[7]);
  dsum = dex + dhy;

}

/* ---------------------------------------------------------------------- */

FixAdress::~FixAdress()
{
  return;
}

/* ---------------------------------------------------------------------- */

int FixAdress::setmask()
{
  int mask = 0;
  mask |= POST_INTEGRATE;
  mask |= PRE_FORCE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixAdress::post_integrate()
{
  // get updated coordinates for atomistic ghosts

  comm->forward_comm_fix(this);

  // clear centres-of-mass of all sites and
  // clear positions of coarse-grained sites

  int *mask = atom->mask;
  double **adw = atom->adw;
  double **x = atom->x;
  int nlocal = atom->nlocal;
  int nghost = atom->nghost;
  int i;
  int newton_bond = force->newton_bond;

  for (i = 0; i < nlocal+nghost; i++) {
    if (mask[i] & allbit) {
      adw[i][0] = adw[i][1] = adw[i][2] = adw[i][3] = 0.0;
    }
    if (mask[i] & coarsebit) {
      x[i][0] = x[i][1] = x[i][2] = 0.0;
    }
  }

  // accumulate positions of coarse-grained sites from directly bonded
  // atomistic sites

  double *mass = atom->mass;
  double m1, m2;
  int g1, g2;
  int **bondlist = neighbor->bondlist;
  int nbondlist = neighbor->nbondlist;
  int i1, i2, n;

  for (n = 0; n < nbondlist; n++) {
    if (bondlist[n][2] != bondtype) continue;
    i1 = bondlist[n][0];
    i2 = bondlist[n][1];
    g1 = mask[i1];
    g2 = mask[i2];
    if (g2 & coarsebit) {
      std::swap(i1,i2);
      std::swap(g1,g2);
    }
    m1 = mass[atom->type[i1]];
    m2 = mass[atom->type[i2]];

    if (newton_bond || i1 < nlocal) {
      if (i1 > nlocal) error->all(FLERR,"i2 > nlocal"); 
      x[i1][0] += x[i2][0]*m2/m1;
      x[i1][1] += x[i2][1]*m2/m1;
      x[i1][2] += x[i2][2]*m2/m1;
    }
  }

  // copy positions to centres-of-mass for coarse-grained sites

  for (i = 0; i < nlocal; i++) {
    if (mask[i] & coarsebit) {
      adw[i][0] = x[i][0];
      adw[i][1] = x[i][1];
      adw[i][2] = x[i][2];
    }
  }

  // calculate adress weights---style specific

  adress_weight();

  return;
}

/* ---------------------------------------------------------------------- */

void FixAdress::pre_force(int /*vflag*/)
{
  // copy adress centres-of-mass and weights from coarse-grained to
  // atomistic sites

  int *mask = atom->mask;
  double **adw = atom->adw;
  int nlocal = atom->nlocal;
  int newton_bond = force->newton_bond;
  int g1, g2;
  int **bondlist = neighbor->bondlist;
  int nbondlist = neighbor->nbondlist;
  int i1, i2, n;

  for (n = 0; n < nbondlist; n++) {
    if (bondlist[n][2] != bondtype) continue;
    i1 = bondlist[n][0];
    i2 = bondlist[n][1];
    g1 = mask[i1];
    g2 = mask[i2];
    if (g2 & coarsebit) {
      std::swap(i1,i2);
      std::swap(g1,g2);
    }
    if (newton_bond || i2 < nlocal) {
      adw[i2][0] += adw[i1][0];
      adw[i2][1] += adw[i1][1];
      adw[i2][2] += adw[i1][2];
      adw[i2][3] += adw[i1][3];
    }
  }

  if (newton_bond) {
    comm->reverse_comm_fix(this);
  }

  return;
}

/* ---------------------------------------------------------------------- */

int FixAdress::pack_forward_comm(int n, int *list, double *buf,
                             int pbc_flag, int * pbc)
{
  int i,j,m;
  double dx,dy,dz;
  int *mask = atom->mask;
  double **x = atom->x;

  m = 0;
  if (pbc_flag == 0) {
    for (i = 0; i < n; i++) {
      j = list[i];
      if (mask[j] & atomisticbit) {
        buf[m++] = x[j][0];
        buf[m++] = x[j][1];
        buf[m++] = x[j][2];
      }
    }
  } else {
    if (domain->triclinic == 0) {
      dx = pbc[0]*domain->xprd;
      dy = pbc[1]*domain->yprd;
      dz = pbc[2]*domain->zprd;
    } else {
      dx = pbc[0]*domain->xprd + pbc[5]*domain->xy + pbc[4]*domain->xz;
      dy = pbc[1]*domain->yprd + pbc[3]*domain->yz;
      dz = pbc[2]*domain->zprd;
    }
    for (i = 0; i < n; i++) {
      j = list[i];
      if (mask[j] & atomisticbit) {
        buf[m++] = x[j][0] + dx;
        buf[m++] = x[j][1] + dy;
        buf[m++] = x[j][2] + dz;
      }
    }
  }

  return m;
}

/* ---------------------------------------------------------------------- */

void FixAdress::unpack_forward_comm(int n, int first, double *buf)
{
  int i,m,last;
  int *mask = atom->mask;
  double **x = atom->x;

  m = 0;
  last = first + n;
  for (i = first; i < last; i++) {
    if (mask[i] & atomisticbit) {
      x[i][0] = buf[m++];
      x[i][1] = buf[m++];
      x[i][2] = buf[m++];
    }
  }
}

/* ---------------------------------------------------------------------- */

int FixAdress::pack_reverse_comm(int n, int first, double *buf)
{
  int i,m,last;
  int *mask = atom->mask;
  double **adw = atom->adw;

  m = 0;
  last = first + n;
  for (i = first; i < last; i++) {
    if (mask[i] & atomisticbit) {
      buf[m++] = adw[i][0];
      buf[m++] = adw[i][1];
      buf[m++] = adw[i][2];
    }
  }

  return m;
}

/* ---------------------------------------------------------------------- */

void FixAdress::unpack_reverse_comm(int n, int *list, double *buf)
{
  int i,j,m;
  int *mask = atom->mask;
  double **adw = atom->adw;

  m = 0;
  for (i = 0; i < n; i++) {
    j = list[i];
    if (mask[j] & atomisticbit) {
      adw[j][0] += buf[m++];
      adw[j][1] += buf[m++];
      adw[j][2] += buf[m++];
    }
  }
}

