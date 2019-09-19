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
#include <cstring>
#include "atom.h"
#include "comm.h"
#include "update.h"
#include "modify.h"
#include "domain.h"
#include "lattice.h"
#include "input.h"
#include "variable.h"
#include "error.h"
#include "force.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "memory.h"

using namespace LAMMPS_NS;
using namespace FixConst;

/* ---------------------------------------------------------------------- */

FixAdress::FixAdress(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg)
{
  comm_reverse = 3;

  // parse args

  if (narg != 6) error->all(FLERR,"Illegal fix adress command");

  bondtype = force->inumeric(FLERR,arg[3]);
  dex = force->numeric(FLERR,arg[4]);
  dhy = force->numeric(FLERR,arg[5]);
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
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixAdress::post_integrate()
{
  // clear positions of coarse-grained sites

  int *res = atom->res;
  double **x = atom->x;
  int nlocal = atom->nlocal;
  int i;

  for (i = 0; i < nlocal; i++) {
    if (res[i] == 0) {
      x[i][0] = 0.0;
      x[i][1] = 0.0;
      x[i][2] = 0.0;
    }
  }

  // accumulate positions of coarse-grained sites from directly bonded
  // atomistic sites

  double *mass = atom->mass;
  double m1, m2;
  int **bondlist = neighbor->bondlist;
  int nbondlist = neighbor->nbondlist;
  int i1, i2, n;

  for (n = 0; n < nbondlist; n++) {
    if (bondlist[n][2] != bondtype) continue;
    i1 = bondlist[n][0];
    i2 = bondlist[n][1];
    if (res[i1] == res[i2]) {
      error->all(FLERR,"the resolutions of the atoms in a bonding pair \
              must differ");
    }
    m1 = mass[atom->type[i1]];
    m2 = mass[atom->type[i2]];
    // coarse-grained site must be local
    if ((res[i1]==0)){
      if (i1 > nlocal) error->all(FLERR,"low resolution atom is a ghost!");
      x[i1][0] += x[i2][0]*m2/m1;
      x[i1][1] += x[i2][1]*m2/m1;
      x[i1][2] += x[i2][2]*m2/m1;
    }
    else if ((res[i2]==0)){
      if (i2 > nlocal) error->all(FLERR,"low resolution atom is a ghost!");
      x[i2][0] += x[i1][0]*m1/m2;
      x[i2][1] += x[i1][1]*m1/m2;
      x[i2][2] += x[i1][2]*m1/m2;
    }
  }

  // clear centres-of-mass of all sites

  double **adw = atom->adw;
  int nghost = atom->nghost;

  for (i = 0; i < nlocal+nghost; i++) {
    adw[i][0] = 0.0;
    adw[i][1] = 0.0;
    adw[i][2] = 0.0;
    adw[i][3] = 0.0;
  }
  // copy positions to centres-of-mass for coarse-grained sites
  for (i = 0; i < nlocal; i++) {
    if (res[i] == 0) {
      adw[i][0] = x[i][0];
      adw[i][1] = x[i][1];
      adw[i][2] = x[i][2];
    }
  }
  // calculate adress weights---style specific
  adress_weight();
  // accumulate centres-of-mass and weights for atomistic particles
  for (n = 0; n < nbondlist; n++) {
    if (bondlist[n][2] != bondtype) continue;
    i1 = bondlist[n][0];
    i2 = bondlist[n][1];
    if ((res[i1]==0)){
      adw[i2][0] += adw[i1][0];
      adw[i2][1] += adw[i1][1];
      adw[i2][2] += adw[i1][2];
      adw[i2][3] += adw[i1][3];
    }
    else if ((res[i2]==0)){
      adw[i1][0] += adw[i2][0];
      adw[i1][1] += adw[i2][1];
      adw[i1][2] += adw[i2][2];
      adw[i1][3] += adw[i2][3];
    }
  }
  comm->reverse_comm_fix(this);

  return;
}

/* ---------------------------------------------------------------------- */

int FixAdress::pack_reverse_comm(int n, int first, double *buf)
{
  int i,m,last;
  double **adw = atom->adw;

  m = 0;
  last = first + n;
  for (i = first; i < last; i++) {
    buf[m++] = adw[i][0];
    buf[m++] = adw[i][1];
    buf[m++] = adw[i][2];
  }
  return m;
}

/* ---------------------------------------------------------------------- */

void FixAdress::unpack_reverse_comm(int n, int *list, double *buf)
{
  int i,j,m;
  double **adw = atom->adw;

  m = 0;
  for (i = 0; i < n; i++) {
    j = list[i];
    adw[j][0] += buf[m++];
    adw[j][1] += buf[m++];
    adw[j][2] += buf[m++];
  }
}

