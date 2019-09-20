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
  comm_reverse = 3;
  commbit = 0;

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
  // clear centres-of-mass of all sites

  int *mask = atom->mask;
  double **adw = atom->adw;
  int nlocal = atom->nlocal;
  int nghost = atom->nghost;
  int i;
  double dx = domain->xprd;
  double dy = domain->yprd;
  double dz = domain->zprd;
  int newton_bond = force->newton_bond;

  for (i = 0; i < nlocal+nghost; i++) {
    if (mask[i] & allbit) {
      adw[i][0] = 0.0;
      adw[i][1] = 0.0;
      adw[i][2] = 0.0;
      adw[i][3] = 0.0;
    }
  }

  // clear positions of coarse-grained sites

  double **x = atom->x;

  for (i = 0; i < nlocal+nghost; i++) {
    if (mask[i] & coarsebit) {
      adw[i][0] = dx*floor(x[i][0]/dx);
      adw[i][1] = dy*floor(x[i][1]/dy);
      adw[i][2] = dz*floor(x[i][2]/dz);
      x[i][0] = 0.0;
      x[i][1] = 0.0;
      x[i][2] = 0.0;
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
    if (g1 == g2) {
      error->all(FLERR,"the resolutions of the atoms in a bonding pair \
              must differ");
    }
    m1 = mass[atom->type[i1]];
    m2 = mass[atom->type[i2]];

    // coarse-grained site must be local

    if ((g1 & coarsebit) && (newton_bond || i1 < nlocal)) {
      x[i1][0] += (x[i2][0]-adw[i1][0])*m2/m1;
      x[i1][1] += (x[i2][1]-adw[i1][1])*m2/m1;
      x[i1][2] += (x[i2][2]-adw[i1][2])*m2/m1;
    }
    else if ((g2 & coarsebit) && (newton_bond || i2 < nlocal)) {
      x[i2][0] += (x[i1][0]-adw[i2][0])*m1/m2;
      x[i2][1] += (x[i1][1]-adw[i2][1])*m1/m2;
      x[i2][2] += (x[i1][2]-adw[i2][2])*m1/m2;
    }
  }
  if (newton_bond) {
    commbit = coarsebit;
    comm->reverse_comm_fix(this);
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
    if ((g1 & coarsebit) && (newton_bond || i2 < nlocal)) {
      adw[i2][0] += adw[i1][0];
      adw[i2][1] += adw[i1][1];
      adw[i2][2] += adw[i1][2];
      adw[i2][3] += adw[i1][3];
    }
    if ((g2 & coarsebit) && (newton_bond || i1 < nlocal)) {
      adw[i1][0] += adw[i2][0];
      adw[i1][1] += adw[i2][1];
      adw[i1][2] += adw[i2][2];
      adw[i1][3] += adw[i2][3];
    }
  }

  if (newton_bond) {
    commbit = atomisticbit;
    comm->reverse_comm_fix(this);
  }

  return;
}

/* ---------------------------------------------------------------------- */

int FixAdress::pack_reverse_comm(int n, int first, double *buf)
{
  int i,m,last;
  int *mask = atom->mask;
  double **arr = NULL;

  if (commbit == atomisticbit) {
    arr = atom->adw;
  } else if (commbit == coarsebit) {
    arr = atom->x;
  }

  m = 0;
  last = first + n;
  for (i = first; i < last; i++) {
    if (mask[i] & commbit) {
      buf[m++] = arr[i][0];
      buf[m++] = arr[i][1];
      buf[m++] = arr[i][2];
    }
  }
  return m;
}

/* ---------------------------------------------------------------------- */

void FixAdress::unpack_reverse_comm(int n, int *list, double *buf)
{
  int i,j,m;
  int *mask = atom->mask;
  double **arr = NULL;

  if (commbit == atomisticbit) {
    arr = atom->adw;
  } else if (commbit == coarsebit) {
    arr = atom->x;
  }

  m = 0;
  for (i = 0; i < n; i++) {
    j = list[i];
    if (mask[j] & commbit) {
      arr[j][0] += buf[m++];
      arr[j][1] += buf[m++];
      arr[j][2] += buf[m++];
    }
  }
}

