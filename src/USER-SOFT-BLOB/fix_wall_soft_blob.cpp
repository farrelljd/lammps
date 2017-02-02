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
#include "fix_wall_soft_blob.h"
#include "atom.h"
#include "error.h"

using namespace LAMMPS_NS;
using namespace FixConst;

/* ---------------------------------------------------------------------- */

FixWallSoftBlob::FixWallSoftBlob(LAMMPS *lmp, int narg, char **arg) :
  FixWall(lmp, narg, arg) {}

/* ---------------------------------------------------------------------- */

void FixWallSoftBlob::precompute(int m)
{
  offset[m] = 3.20 * exp(-4.17 * (cutoff[m] - 0.50));
}

/* ----------------------------------------------------------------------
   interaction of all particles in group with a wall
   m = index of wall coeffs
   which = xlo,xhi,ylo,yhi,zlo,zhi
   error if any particle is on or behind wall
------------------------------------------------------------------------- */

void FixWallSoftBlob::wall_particle(int m, int which, double coord)
{
  double delta,fwall;

  double **x = atom->x;
  double **f = atom->f;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  int dim = which / 2;
  int side = which % 2;
  if (side == 0) side = -1;

  int onflag = 0;

  double energy;
  for (int i = 0; i < nlocal; i++)
    if (mask[i] & groupbit) {
      if (side < 0) delta = x[i][dim] - coord;
      else delta = coord - x[i][dim];
      if (delta >= cutoff[m]) continue;
      if (delta <= 0.0) {
        onflag = 1;
        continue;
      }
      energy = 3.20 * exp(-4.17 * (delta - 0.50));
      fwall = side * 4.17 * energy;
      f[i][dim] -= fwall;
      ewall[0] += energy - offset[m];
      ewall[m+1] += fwall;
    }

  if (onflag) error->one(FLERR,"Particle on or inside fix wall surface");
}
