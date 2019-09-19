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

#include "fix_adress_sphere.h"
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
#include "math.h"
#include "math_const.h"

using namespace LAMMPS_NS;
using namespace FixConst;
using namespace MathConst;

/* ---------------------------------------------------------------------- */

FixAdressSphere::FixAdressSphere(LAMMPS *lmp, int narg, char **arg) :
  FixAdress(lmp, narg, arg)
{
  factor = MY_PI/2/dhy;
}

/* ---------------------------------------------------------------------- */

void FixAdressSphere::adress_weight()
{
  int *mask = atom->mask;
  double **x = atom->x;
  int *res = atom->res;
  double **adw = atom->adw;
  double cx, cy, cz, cd;
  double c0x = domain->xprd_half;
  double c0y = domain->yprd_half;
  double c0z = domain->zprd_half;
  int i;
  int nlocal = atom->nlocal;

  for (i = 0; i < nlocal; i++) {
    if (mask[i] & coarsebit) {
      cx = c0x - x[i][0];
      cy = c0y - x[i][1];
      cz = c0z - x[i][2];
      cd = sqrt(cx*cx + cy*cy + cz*cz);
      if (cd > dsum) {
        adw[i][3] = 1.0;
      } else if (cd < dex) {
        adw[i][3] = 1.0;
      } else {
        adw[i][3] = cos(factor*(cd-dex));
        adw[i][3]*= adw[i][3];
      }
    }
  }
}
