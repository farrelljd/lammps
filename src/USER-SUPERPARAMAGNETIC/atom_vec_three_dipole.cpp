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
#include "atom.h"
#include "atom_vec_three_dipole.h"
#include "comm.h"
#include "domain.h"
#include "modify.h"
#include "fix.h"
#include "memory.h"
#include "error.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

AtomVecThreeDipole::AtomVecThreeDipole(LAMMPS *lmp) : AtomVec(lmp)
{
  molecular = 0;
  mass_type = 1;

  comm_x_only = 0;
  comm_f_only = 1;
  size_forward = 12;
  size_reverse = 3;
  size_border = 19;//+8
  size_velocity = 3;
  size_data_atom = 15;
  size_data_vel = 4;
  xcol_data = 4;

  atom->q_flag = atom->mu_flag = 1;
  atom->mu_x_flag = atom->mu_y_flag = atom->mu_z_flag = 1;
}

/* ----------------------------------------------------------------------
   grow atom arrays
   n = 0 grows arrays by a chunk
   n > 0 allocates arrays to size n
------------------------------------------------------------------------- */

void AtomVecThreeDipole::grow(int n)
{
  if (n == 0) grow_nmax();
  else nmax = n;
  atom->nmax = nmax;
  if (nmax < 0 || nmax > MAXSMALLINT)
    error->one(FLERR,"Per-processor system is too big");

  tag = memory->grow(atom->tag,nmax,"atom:tag");
  type = memory->grow(atom->type,nmax,"atom:type");
  mask = memory->grow(atom->mask,nmax,"atom:mask");
  image = memory->grow(atom->image,nmax,"atom:image");
  x = memory->grow(atom->x,nmax,3,"atom:x");
  v = memory->grow(atom->v,nmax,3,"atom:v");
  f = memory->grow(atom->f,nmax*comm->nthreads,3,"atom:f");

  q = memory->grow(atom->q,nmax,"atom:q");
  mu = memory->grow(atom->mu,nmax,4,"atom:mu");
  mu_x = memory->grow(atom->mu_x,nmax,4,"atom:mu_x");
  mu_y = memory->grow(atom->mu_y,nmax,4,"atom:mu_y");
  mu_z = memory->grow(atom->mu_z,nmax,4,"atom:mu_z");

  if (atom->nextra_grow)
    for (int iextra = 0; iextra < atom->nextra_grow; iextra++)
      modify->fix[atom->extra_grow[iextra]]->grow_arrays(nmax);
}

/* ----------------------------------------------------------------------
   reset local array ptrs
------------------------------------------------------------------------- */

void AtomVecThreeDipole::grow_reset()
{
  tag = atom->tag; type = atom->type;
  mask = atom->mask; image = atom->image;
  x = atom->x; v = atom->v; f = atom->f;
  q = atom->q; mu = atom->mu;
  mu_x = atom->mu_x; mu_y = atom->mu_y; mu_z = atom->mu_z;
}

/* ----------------------------------------------------------------------
   copy atom I info to atom J
------------------------------------------------------------------------- */

void AtomVecThreeDipole::copy(int i, int j, int delflag)
{
  tag[j] = tag[i];
  type[j] = type[i];
  mask[j] = mask[i];
  image[j] = image[i];
  x[j][0] = x[i][0];
  x[j][1] = x[i][1];
  x[j][2] = x[i][2];
  v[j][0] = v[i][0];
  v[j][1] = v[i][1];
  v[j][2] = v[i][2];

  q[j] = q[i];
  mu_x[j][0] = mu_x[i][0];
  mu_x[j][1] = mu_x[i][1];
  mu_x[j][2] = mu_x[i][2];
  mu_x[j][3] = mu_x[i][3];
  mu_y[j][0] = mu_y[i][0];
  mu_y[j][1] = mu_y[i][1];
  mu_y[j][2] = mu_y[i][2];
  mu_y[j][3] = mu_y[i][3];
  mu_z[j][0] = mu_z[i][0];
  mu_z[j][1] = mu_z[i][1];
  mu_z[j][2] = mu_z[i][2];
  mu_z[j][3] = mu_z[i][3];

  if (atom->nextra_grow)
    for (int iextra = 0; iextra < atom->nextra_grow; iextra++)
      modify->fix[atom->extra_grow[iextra]]->copy_arrays(i,j,delflag);
}

/* ---------------------------------------------------------------------- */

int AtomVecThreeDipole::pack_comm(int n, int *list, double *buf,
                             int pbc_flag, int *pbc)
{
  int i,j,m;
  double dx,dy,dz;

  m = 0;
  if (pbc_flag == 0) {
    for (i = 0; i < n; i++) {
      j = list[i];
      buf[m++] = x[j][0];
      buf[m++] = x[j][1];
      buf[m++] = x[j][2];
      buf[m++] = mu_x[j][0];
      buf[m++] = mu_x[j][1];
      buf[m++] = mu_x[j][2];
      buf[m++] = mu_y[j][0];
      buf[m++] = mu_y[j][1];
      buf[m++] = mu_y[j][2];
      buf[m++] = mu_z[j][0];
      buf[m++] = mu_z[j][1];
      buf[m++] = mu_z[j][2];
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
      buf[m++] = x[j][0] + dx;
      buf[m++] = x[j][1] + dy;
      buf[m++] = x[j][2] + dz;
      buf[m++] = mu_x[j][0];
      buf[m++] = mu_x[j][1];
      buf[m++] = mu_x[j][2];
      buf[m++] = mu_y[j][0];
      buf[m++] = mu_y[j][1];
      buf[m++] = mu_y[j][2];
      buf[m++] = mu_z[j][0];
      buf[m++] = mu_z[j][1];
      buf[m++] = mu_z[j][2];
    }
  }
  return m;
}

/* ---------------------------------------------------------------------- */

int AtomVecThreeDipole::pack_comm_vel(int n, int *list, double *buf,
                                 int pbc_flag, int *pbc)
{
  int i,j,m;
  double dx,dy,dz,dvx,dvy,dvz;

  m = 0;
  if (pbc_flag == 0) {
    for (i = 0; i < n; i++) {
      j = list[i];
      buf[m++] = x[j][0];
      buf[m++] = x[j][1];
      buf[m++] = x[j][2];
      buf[m++] = mu_x[j][0];
      buf[m++] = mu_x[j][1];
      buf[m++] = mu_x[j][2];
      buf[m++] = mu_y[j][0];
      buf[m++] = mu_y[j][1];
      buf[m++] = mu_y[j][2];
      buf[m++] = mu_z[j][0];
      buf[m++] = mu_z[j][1];
      buf[m++] = mu_z[j][2];
      buf[m++] = v[j][0];
      buf[m++] = v[j][1];
      buf[m++] = v[j][2];
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
    if (!deform_vremap) {
      for (i = 0; i < n; i++) {
        j = list[i];
        buf[m++] = x[j][0] + dx;
        buf[m++] = x[j][1] + dy;
        buf[m++] = x[j][2] + dz;
        buf[m++] = mu_x[j][0];
        buf[m++] = mu_x[j][1];
        buf[m++] = mu_x[j][2];
        buf[m++] = mu_y[j][0];
        buf[m++] = mu_y[j][1];
        buf[m++] = mu_y[j][2];
        buf[m++] = mu_z[j][0];
        buf[m++] = mu_z[j][1];
        buf[m++] = mu_z[j][2];
        buf[m++] = v[j][0];
        buf[m++] = v[j][1];
        buf[m++] = v[j][2];
      }
    } else {
      dvx = pbc[0]*h_rate[0] + pbc[5]*h_rate[5] + pbc[4]*h_rate[4];
      dvy = pbc[1]*h_rate[1] + pbc[3]*h_rate[3];
      dvz = pbc[2]*h_rate[2];
      for (i = 0; i < n; i++) {
        j = list[i];
        buf[m++] = x[j][0] + dx;
        buf[m++] = x[j][1] + dy;
        buf[m++] = x[j][2] + dz;
        buf[m++] = mu_x[j][0];
        buf[m++] = mu_x[j][1];
        buf[m++] = mu_x[j][2];
        buf[m++] = mu_y[j][0];
        buf[m++] = mu_y[j][1];
        buf[m++] = mu_y[j][2];
        buf[m++] = mu_z[j][0];
        buf[m++] = mu_z[j][1];
        buf[m++] = mu_z[j][2];
        if (mask[i] & deform_groupbit) {
          buf[m++] = v[j][0] + dvx;
          buf[m++] = v[j][1] + dvy;
          buf[m++] = v[j][2] + dvz;
        } else {
          buf[m++] = v[j][0];
          buf[m++] = v[j][1];
          buf[m++] = v[j][2];
        }
      }
    }
  }
  return m;
}

/* ---------------------------------------------------------------------- */

int AtomVecThreeDipole::pack_comm_hybrid(int n, int *list, double *buf)
{
  int i,j,m;

  m = 0;
  for (i = 0; i < n; i++) {
    j = list[i];
    buf[m++] = mu_x[j][0];
    buf[m++] = mu_x[j][1];
    buf[m++] = mu_x[j][2];
    buf[m++] = mu_y[j][0];
    buf[m++] = mu_y[j][1];
    buf[m++] = mu_y[j][2];
    buf[m++] = mu_z[j][0];
    buf[m++] = mu_z[j][1];
    buf[m++] = mu_z[j][2];
  }
  return m;
}

/* ---------------------------------------------------------------------- */

void AtomVecThreeDipole::unpack_comm(int n, int first, double *buf)
{
  int i,m,last;

  m = 0;
  last = first + n;
  for (i = first; i < last; i++) {
    x[i][0] = buf[m++];
    x[i][1] = buf[m++];
    x[i][2] = buf[m++];
    mu_x[i][0] = buf[m++];
    mu_x[i][1] = buf[m++];
    mu_x[i][2] = buf[m++];
    mu_y[i][0] = buf[m++];
    mu_y[i][1] = buf[m++];
    mu_y[i][2] = buf[m++];
    mu_z[i][0] = buf[m++];
    mu_z[i][1] = buf[m++];
    mu_z[i][2] = buf[m++];
  }
}

/* ---------------------------------------------------------------------- */

void AtomVecThreeDipole::unpack_comm_vel(int n, int first, double *buf)
{
  int i,m,last;

  m = 0;
  last = first + n;
  for (i = first; i < last; i++) {
    x[i][0] = buf[m++];
    x[i][1] = buf[m++];
    x[i][2] = buf[m++];
    mu_x[i][0] = buf[m++];
    mu_x[i][1] = buf[m++];
    mu_x[i][2] = buf[m++];
    mu_y[i][0] = buf[m++];
    mu_y[i][1] = buf[m++];
    mu_y[i][2] = buf[m++];
    mu_z[i][0] = buf[m++];
    mu_z[i][1] = buf[m++];
    mu_z[i][2] = buf[m++];
    v[i][0] = buf[m++];
    v[i][1] = buf[m++];
    v[i][2] = buf[m++];
  }
}

/* ---------------------------------------------------------------------- */

int AtomVecThreeDipole::unpack_comm_hybrid(int n, int first, double *buf)
{
  int i,m,last;

  m = 0;
  last = first + n;
  for (i = first; i < last; i++) {
    mu_x[i][0] = buf[m++];
    mu_x[i][1] = buf[m++];
    mu_x[i][2] = buf[m++];
    mu_y[i][0] = buf[m++];
    mu_y[i][1] = buf[m++];
    mu_y[i][2] = buf[m++];
    mu_z[i][0] = buf[m++];
    mu_z[i][1] = buf[m++];
    mu_z[i][2] = buf[m++];
  }
  return m;
}

/* ---------------------------------------------------------------------- */

int AtomVecThreeDipole::pack_reverse(int n, int first, double *buf)
{
  int i,m,last;

  m = 0;
  last = first + n;
  for (i = first; i < last; i++) {
    buf[m++] = f[i][0];
    buf[m++] = f[i][1];
    buf[m++] = f[i][2];
  }
  return m;
}

/* ---------------------------------------------------------------------- */

void AtomVecThreeDipole::unpack_reverse(int n, int *list, double *buf)
{
  int i,j,m;

  m = 0;
  for (i = 0; i < n; i++) {
    j = list[i];
    f[j][0] += buf[m++];
    f[j][1] += buf[m++];
    f[j][2] += buf[m++];
  }
}

/* ---------------------------------------------------------------------- */

int AtomVecThreeDipole::pack_border(int n, int *list, double *buf,
                               int pbc_flag, int *pbc)
{
  int i,j,m;
  double dx,dy,dz;

  m = 0;
  if (pbc_flag == 0) {
    for (i = 0; i < n; i++) {
      j = list[i];
      buf[m++] = x[j][0];
      buf[m++] = x[j][1];
      buf[m++] = x[j][2];
      buf[m++] = ubuf(tag[j]).d;
      buf[m++] = ubuf(type[j]).d;
      buf[m++] = ubuf(mask[j]).d;
      buf[m++] = q[j];
      buf[m++] = mu_x[j][0];
      buf[m++] = mu_x[j][1];
      buf[m++] = mu_x[j][2];
      buf[m++] = mu_x[j][3];
      buf[m++] = mu_y[j][0];
      buf[m++] = mu_y[j][1];
      buf[m++] = mu_y[j][2];
      buf[m++] = mu_y[j][3];
      buf[m++] = mu_z[j][0];
      buf[m++] = mu_z[j][1];
      buf[m++] = mu_z[j][2];
      buf[m++] = mu_z[j][3];
    }
  } else {
    if (domain->triclinic == 0) {
      dx = pbc[0]*domain->xprd;
      dy = pbc[1]*domain->yprd;
      dz = pbc[2]*domain->zprd;
    } else {
      dx = pbc[0];
      dy = pbc[1];
      dz = pbc[2];
    }
    for (i = 0; i < n; i++) {
      j = list[i];
      buf[m++] = x[j][0] + dx;
      buf[m++] = x[j][1] + dy;
      buf[m++] = x[j][2] + dz;
      buf[m++] = ubuf(tag[j]).d;
      buf[m++] = ubuf(type[j]).d;
      buf[m++] = ubuf(mask[j]).d;
      buf[m++] = q[j];
      buf[m++] = mu_x[j][0];
      buf[m++] = mu_x[j][1];
      buf[m++] = mu_x[j][2];
      buf[m++] = mu_x[j][3];
      buf[m++] = mu_y[j][0];
      buf[m++] = mu_y[j][1];
      buf[m++] = mu_y[j][2];
      buf[m++] = mu_y[j][3];
      buf[m++] = mu_z[j][0];
      buf[m++] = mu_z[j][1];
      buf[m++] = mu_z[j][2];
      buf[m++] = mu_z[j][3];
    }
  }

  if (atom->nextra_border)
    for (int iextra = 0; iextra < atom->nextra_border; iextra++)
      m += modify->fix[atom->extra_border[iextra]]->pack_border(n,list,&buf[m]);

  return m;
}

/* ---------------------------------------------------------------------- */

int AtomVecThreeDipole::pack_border_vel(int n, int *list, double *buf,
                                   int pbc_flag, int *pbc)
{
  int i,j,m;
  double dx,dy,dz,dvx,dvy,dvz;

  m = 0;
  if (pbc_flag == 0) {
    for (i = 0; i < n; i++) {
      j = list[i];
      buf[m++] = x[j][0];
      buf[m++] = x[j][1];
      buf[m++] = x[j][2];
      buf[m++] = ubuf(tag[j]).d;
      buf[m++] = ubuf(type[j]).d;
      buf[m++] = ubuf(mask[j]).d;
      buf[m++] = q[j];
      buf[m++] = mu_x[j][0];
      buf[m++] = mu_x[j][1];
      buf[m++] = mu_x[j][2];
      buf[m++] = mu_x[j][3];
      buf[m++] = mu_y[j][0];
      buf[m++] = mu_y[j][1];
      buf[m++] = mu_y[j][2];
      buf[m++] = mu_y[j][3];
      buf[m++] = mu_z[j][0];
      buf[m++] = mu_z[j][1];
      buf[m++] = mu_z[j][2];
      buf[m++] = mu_z[j][3];
      buf[m++] = v[j][0];
      buf[m++] = v[j][1];
      buf[m++] = v[j][2];
    }
  } else {
    if (domain->triclinic == 0) {
      dx = pbc[0]*domain->xprd;
      dy = pbc[1]*domain->yprd;
      dz = pbc[2]*domain->zprd;
    } else {
      dx = pbc[0];
      dy = pbc[1];
      dz = pbc[2];
    }
    if (!deform_vremap) {
      for (i = 0; i < n; i++) {
        j = list[i];
        buf[m++] = x[j][0] + dx;
        buf[m++] = x[j][1] + dy;
        buf[m++] = x[j][2] + dz;
        buf[m++] = ubuf(tag[j]).d;
        buf[m++] = ubuf(type[j]).d;
        buf[m++] = ubuf(mask[j]).d;
        buf[m++] = q[j];
        buf[m++] = mu_x[j][0];
        buf[m++] = mu_x[j][1];
        buf[m++] = mu_x[j][2];
        buf[m++] = mu_x[j][3];
        buf[m++] = mu_y[j][0];
        buf[m++] = mu_y[j][1];
        buf[m++] = mu_y[j][2];
        buf[m++] = mu_y[j][3];
        buf[m++] = mu_z[j][0];
        buf[m++] = mu_z[j][1];
        buf[m++] = mu_z[j][2];
        buf[m++] = mu_z[j][3];
        buf[m++] = v[j][0];
        buf[m++] = v[j][1];
        buf[m++] = v[j][2];
      }
    } else {
      dvx = pbc[0]*h_rate[0] + pbc[5]*h_rate[5] + pbc[4]*h_rate[4];
      dvy = pbc[1]*h_rate[1] + pbc[3]*h_rate[3];
      dvz = pbc[2]*h_rate[2];
      for (i = 0; i < n; i++) {
        j = list[i];
        buf[m++] = x[j][0] + dx;
        buf[m++] = x[j][1] + dy;
        buf[m++] = x[j][2] + dz;
        buf[m++] = ubuf(tag[j]).d;
        buf[m++] = ubuf(type[j]).d;
        buf[m++] = ubuf(mask[j]).d;
        buf[m++] = q[j];
        buf[m++] = mu_x[j][0];
        buf[m++] = mu_x[j][1];
        buf[m++] = mu_x[j][2];
        buf[m++] = mu_x[j][3];
        buf[m++] = mu_y[j][0];
        buf[m++] = mu_y[j][1];
        buf[m++] = mu_y[j][2];
        buf[m++] = mu_y[j][3];
        buf[m++] = mu_z[j][0];
        buf[m++] = mu_z[j][1];
        buf[m++] = mu_z[j][2];
        buf[m++] = mu_z[j][3];
        if (mask[i] & deform_groupbit) {
          buf[m++] = v[j][0] + dvx;
          buf[m++] = v[j][1] + dvy;
          buf[m++] = v[j][2] + dvz;
        } else {
          buf[m++] = v[j][0];
          buf[m++] = v[j][1];
          buf[m++] = v[j][2];
        }
      }
    }
  }

  if (atom->nextra_border)
    for (int iextra = 0; iextra < atom->nextra_border; iextra++)
      m += modify->fix[atom->extra_border[iextra]]->pack_border(n,list,&buf[m]);

  return m;
}

/* ---------------------------------------------------------------------- */

int AtomVecThreeDipole::pack_border_hybrid(int n, int *list, double *buf)
{
  int i,j,m;

  m = 0;
  for (i = 0; i < n; i++) {
    j = list[i];
    buf[m++] = q[j];
    buf[m++] = mu_x[j][0];
    buf[m++] = mu_x[j][1];
    buf[m++] = mu_x[j][2];
    buf[m++] = mu_x[j][3];
    buf[m++] = mu_y[j][0];
    buf[m++] = mu_y[j][1];
    buf[m++] = mu_y[j][2];
    buf[m++] = mu_y[j][3];
    buf[m++] = mu_z[j][0];
    buf[m++] = mu_z[j][1];
    buf[m++] = mu_z[j][2];
    buf[m++] = mu_z[j][3];
  }
  return m;
}

/* ---------------------------------------------------------------------- */

void AtomVecThreeDipole::unpack_border(int n, int first, double *buf)
{
  int i,m,last;

  m = 0;
  last = first + n;
  for (i = first; i < last; i++) {
    if (i == nmax) grow(0);
    x[i][0] = buf[m++];
    x[i][1] = buf[m++];
    x[i][2] = buf[m++];
    tag[i] = (tagint) ubuf(buf[m++]).i;
    type[i] = (int) ubuf(buf[m++]).i;
    mask[i] = (int) ubuf(buf[m++]).i;
    q[i] = buf[m++];
    mu_x[i][0] = buf[m++];
    mu_x[i][1] = buf[m++];
    mu_x[i][2] = buf[m++];
    mu_x[i][3] = buf[m++];
    mu_y[i][0] = buf[m++];
    mu_y[i][1] = buf[m++];
    mu_y[i][2] = buf[m++];
    mu_y[i][3] = buf[m++];
    mu_z[i][0] = buf[m++];
    mu_z[i][1] = buf[m++];
    mu_z[i][2] = buf[m++];
    mu_z[i][3] = buf[m++];
  }

  if (atom->nextra_border)
    for (int iextra = 0; iextra < atom->nextra_border; iextra++)
      m += modify->fix[atom->extra_border[iextra]]->
        unpack_border(n,first,&buf[m]);
}

/* ---------------------------------------------------------------------- */

void AtomVecThreeDipole::unpack_border_vel(int n, int first, double *buf)
{
  int i,m,last;

  m = 0;
  last = first + n;
  for (i = first; i < last; i++) {
    if (i == nmax) grow(0);
    x[i][0] = buf[m++];
    x[i][1] = buf[m++];
    x[i][2] = buf[m++];
    tag[i] = (tagint) ubuf(buf[m++]).i;
    type[i] = (int) ubuf(buf[m++]).i;
    mask[i] = (int) ubuf(buf[m++]).i;
    q[i] = buf[m++];
    mu_x[i][0] = buf[m++];
    mu_x[i][1] = buf[m++];
    mu_x[i][2] = buf[m++];
    mu_x[i][3] = buf[m++];
    mu_y[i][0] = buf[m++];
    mu_y[i][1] = buf[m++];
    mu_y[i][2] = buf[m++];
    mu_y[i][3] = buf[m++];
    mu_z[i][0] = buf[m++];
    mu_z[i][1] = buf[m++];
    mu_z[i][2] = buf[m++];
    mu_z[i][3] = buf[m++];
    v[i][0] = buf[m++];
    v[i][1] = buf[m++];
    v[i][2] = buf[m++];
  }

  if (atom->nextra_border)
    for (int iextra = 0; iextra < atom->nextra_border; iextra++)
      m += modify->fix[atom->extra_border[iextra]]->
        unpack_border(n,first,&buf[m]);
}

/* ---------------------------------------------------------------------- */

int AtomVecThreeDipole::unpack_border_hybrid(int n, int first, double *buf)
{
  int i,m,last;

  m = 0;
  last = first + n;
  for (i = first; i < last; i++) {
    q[i] = buf[m++];
    mu_x[i][0] = buf[m++];
    mu_x[i][1] = buf[m++];
    mu_x[i][2] = buf[m++];
    mu_x[i][3] = buf[m++];
    mu_y[i][0] = buf[m++];
    mu_y[i][1] = buf[m++];
    mu_y[i][2] = buf[m++];
    mu_y[i][3] = buf[m++];
    mu_z[i][0] = buf[m++];
    mu_z[i][1] = buf[m++];
    mu_z[i][2] = buf[m++];
    mu_z[i][3] = buf[m++];
  }
  return m;
}

/* ----------------------------------------------------------------------
   pack all atom quantities for shipping to another proc
   xyz must be 1st 3 values, so that comm::exchange can test on them
------------------------------------------------------------------------- */

int AtomVecThreeDipole::pack_exchange(int i, double *buf)
{
  int m = 1;
  buf[m++] = x[i][0];
  buf[m++] = x[i][1];
  buf[m++] = x[i][2];
  buf[m++] = v[i][0];
  buf[m++] = v[i][1];
  buf[m++] = v[i][2];
  buf[m++] = ubuf(tag[i]).d;
  buf[m++] = ubuf(type[i]).d;
  buf[m++] = ubuf(mask[i]).d;
  buf[m++] = ubuf(image[i]).d;

  buf[m++] = q[i];
  buf[m++] = mu_x[i][0];
  buf[m++] = mu_x[i][1];
  buf[m++] = mu_x[i][2];
  buf[m++] = mu_x[i][3];
  buf[m++] = mu_y[i][0];
  buf[m++] = mu_y[i][1];
  buf[m++] = mu_y[i][2];
  buf[m++] = mu_y[i][3];
  buf[m++] = mu_z[i][0];
  buf[m++] = mu_z[i][1];
  buf[m++] = mu_z[i][2];
  buf[m++] = mu_z[i][3];

  if (atom->nextra_grow)
    for (int iextra = 0; iextra < atom->nextra_grow; iextra++)
      m += modify->fix[atom->extra_grow[iextra]]->pack_exchange(i,&buf[m]);

  buf[0] = m;
  return m;
}

/* ---------------------------------------------------------------------- */

int AtomVecThreeDipole::unpack_exchange(double *buf)
{
  int nlocal = atom->nlocal;
  if (nlocal == nmax) grow(0);

  int m = 1;
  x[nlocal][0] = buf[m++];
  x[nlocal][1] = buf[m++];
  x[nlocal][2] = buf[m++];
  v[nlocal][0] = buf[m++];
  v[nlocal][1] = buf[m++];
  v[nlocal][2] = buf[m++];
  tag[nlocal] = (tagint) ubuf(buf[m++]).i;
  type[nlocal] = (int) ubuf(buf[m++]).i;
  mask[nlocal] = (int) ubuf(buf[m++]).i;
  image[nlocal] = (imageint) ubuf(buf[m++]).i;

  q[nlocal] = buf[m++];
  mu_x[nlocal][0] = buf[m++];
  mu_x[nlocal][1] = buf[m++];
  mu_x[nlocal][2] = buf[m++];
  mu_x[nlocal][3] = buf[m++];
  mu_y[nlocal][0] = buf[m++];
  mu_y[nlocal][1] = buf[m++];
  mu_y[nlocal][2] = buf[m++];
  mu_y[nlocal][3] = buf[m++];
  mu_z[nlocal][0] = buf[m++];
  mu_z[nlocal][1] = buf[m++];
  mu_z[nlocal][2] = buf[m++];
  mu_z[nlocal][3] = buf[m++];

  if (atom->nextra_grow)
    for (int iextra = 0; iextra < atom->nextra_grow; iextra++)
      m += modify->fix[atom->extra_grow[iextra]]->
        unpack_exchange(nlocal,&buf[m]);

  atom->nlocal++;
  return m;
}

/* ----------------------------------------------------------------------
   size of restart data for all atoms owned by this proc
   include extra data stored by fixes
------------------------------------------------------------------------- */

int AtomVecThreeDipole::size_restart()
{
  int i;

  int nlocal = atom->nlocal;
  int n = 19 * nlocal;

  if (atom->nextra_restart)
    for (int iextra = 0; iextra < atom->nextra_restart; iextra++)
      for (i = 0; i < nlocal; i++)
        n += modify->fix[atom->extra_restart[iextra]]->size_restart(i);

  return n;
}

/* ----------------------------------------------------------------------
   pack atom I's data for restart file including extra quantities
   xyz must be 1st 3 values, so that read_restart can test on them
   molecular types may be negative, but write as positive
------------------------------------------------------------------------- */

int AtomVecThreeDipole::pack_restart(int i, double *buf)
{
  int m = 1;
  buf[m++] = x[i][0];
  buf[m++] = x[i][1];
  buf[m++] = x[i][2];
  buf[m++] = ubuf(tag[i]).d;
  buf[m++] = ubuf(type[i]).d;
  buf[m++] = ubuf(mask[i]).d;
  buf[m++] = ubuf(image[i]).d;
  buf[m++] = v[i][0];
  buf[m++] = v[i][1];
  buf[m++] = v[i][2];

  buf[m++] = q[i];
  buf[m++] = mu_x[i][0];
  buf[m++] = mu_x[i][1];
  buf[m++] = mu_x[i][2];
  buf[m++] = mu_x[i][3];
  buf[m++] = mu_y[i][0];
  buf[m++] = mu_y[i][1];
  buf[m++] = mu_y[i][2];
  buf[m++] = mu_y[i][3];
  buf[m++] = mu_z[i][0];
  buf[m++] = mu_z[i][1];
  buf[m++] = mu_z[i][2];
  buf[m++] = mu_z[i][3];

  if (atom->nextra_restart)
    for (int iextra = 0; iextra < atom->nextra_restart; iextra++)
      m += modify->fix[atom->extra_restart[iextra]]->pack_restart(i,&buf[m]);

  buf[0] = m;
  return m;
}

/* ----------------------------------------------------------------------
   unpack data for one atom from restart file including extra quantities
------------------------------------------------------------------------- */

int AtomVecThreeDipole::unpack_restart(double *buf)
{
  int nlocal = atom->nlocal;
  if (nlocal == nmax) {
    grow(0);
    if (atom->nextra_store)
      memory->grow(atom->extra,nmax,atom->nextra_store,"atom:extra");
  }

  int m = 1;
  x[nlocal][0] = buf[m++];
  x[nlocal][1] = buf[m++];
  x[nlocal][2] = buf[m++];
  tag[nlocal] = (tagint) ubuf(buf[m++]).i;
  type[nlocal] = (int) ubuf(buf[m++]).i;
  mask[nlocal] = (int) ubuf(buf[m++]).i;
  image[nlocal] = (imageint) ubuf(buf[m++]).i;
  v[nlocal][0] = buf[m++];
  v[nlocal][1] = buf[m++];
  v[nlocal][2] = buf[m++];

  q[nlocal] = buf[m++];
  mu_x[nlocal][0] = buf[m++];
  mu_x[nlocal][1] = buf[m++];
  mu_x[nlocal][2] = buf[m++];
  mu_x[nlocal][3] = buf[m++];
  mu_y[nlocal][0] = buf[m++];
  mu_y[nlocal][1] = buf[m++];
  mu_y[nlocal][2] = buf[m++];
  mu_y[nlocal][3] = buf[m++];
  mu_z[nlocal][0] = buf[m++];
  mu_z[nlocal][1] = buf[m++];
  mu_z[nlocal][2] = buf[m++];
  mu_z[nlocal][3] = buf[m++];

  double **extra = atom->extra;
  if (atom->nextra_store) {
    int size = static_cast<int> (buf[0]) - m;
    for (int i = 0; i < size; i++) extra[nlocal][i] = buf[m++];
  }

  atom->nlocal++;
  return m;
}

/* ----------------------------------------------------------------------
   create one atom of itype at coord
   set other values to defaults
------------------------------------------------------------------------- */

void AtomVecThreeDipole::create_atom(int itype, double *coord)
{
  int nlocal = atom->nlocal;
  if (nlocal == nmax) grow(0);

  tag[nlocal] = 0;
  type[nlocal] = itype;
  x[nlocal][0] = coord[0];
  x[nlocal][1] = coord[1];
  x[nlocal][2] = coord[2];
  mask[nlocal] = 1;
  image[nlocal] = ((imageint) IMGMAX << IMG2BITS) |
    ((imageint) IMGMAX << IMGBITS) | IMGMAX;
  v[nlocal][0] = 0.0;
  v[nlocal][1] = 0.0;
  v[nlocal][2] = 0.0;

  q[nlocal] = 0.0;
  mu[nlocal][0] = 0.0;
  mu[nlocal][1] = 0.0;
  mu[nlocal][2] = 0.0;
  mu[nlocal][3] = 0.0;
  mu_x[nlocal][0] = 1.0;
  mu_x[nlocal][1] = 0.0;
  mu_x[nlocal][2] = 0.0;
  mu_x[nlocal][3] = 1.0;
  mu_y[nlocal][0] = 0.0;
  mu_y[nlocal][1] = 1.0;
  mu_y[nlocal][2] = 0.0;
  mu_y[nlocal][3] = 1.0;
  mu_z[nlocal][0] = 0.0;
  mu_z[nlocal][1] = 0.0;
  mu_z[nlocal][2] = 1.0;
  mu_z[nlocal][3] = 1.0;

  atom->nlocal++;
}

/* ----------------------------------------------------------------------
   unpack one line from Atoms section of data file
   initialize other atom quantities
------------------------------------------------------------------------- */

void AtomVecThreeDipole::data_atom(double *coord, imageint imagetmp, char **values)
{
  int nlocal = atom->nlocal;
  if (nlocal == nmax) grow(0);

  tag[nlocal] = ATOTAGINT(values[0]);
  type[nlocal] = atoi(values[1]);
  if (type[nlocal] <= 0 || type[nlocal] > atom->ntypes)
    error->one(FLERR,"Invalid atom type in Atoms section of data file");

  q[nlocal] = atof(values[2]);

  x[nlocal][0] = coord[0];
  x[nlocal][1] = coord[1];
  x[nlocal][2] = coord[2];

  mu_x[nlocal][0] = atof(values[6]);
  mu_x[nlocal][1] = atof(values[7]);
  mu_x[nlocal][2] = atof(values[8]);
  mu_x[nlocal][3] = sqrt(mu_x[nlocal][0]*mu_x[nlocal][0] +
                       mu_x[nlocal][1]*mu_x[nlocal][1] +
                       mu_x[nlocal][2]*mu_x[nlocal][2]);
  mu_y[nlocal][0] = atof(values[9]);
  mu_y[nlocal][1] = atof(values[10]);
  mu_y[nlocal][2] = atof(values[11]);
  mu_y[nlocal][3] = sqrt(mu_y[nlocal][0]*mu_y[nlocal][0] +
                       mu_y[nlocal][1]*mu_y[nlocal][1] +
                       mu_y[nlocal][2]*mu_y[nlocal][2]);
  mu_z[nlocal][0] = atof(values[12]);
  mu_z[nlocal][1] = atof(values[13]);
  mu_z[nlocal][2] = atof(values[14]);
  mu_z[nlocal][3] = sqrt(mu_z[nlocal][0]*mu_z[nlocal][0] +
                        mu_z[nlocal][1]*mu_z[nlocal][1] +
                        mu_z[nlocal][2]*mu_z[nlocal][2]);

  image[nlocal] = imagetmp;

  mask[nlocal] = 1;
  v[nlocal][0] = 0.0;
  v[nlocal][1] = 0.0;
  v[nlocal][2] = 0.0;

  atom->nlocal++;
}

/* ----------------------------------------------------------------------
   unpack hybrid quantities from one line in Atoms section of data file
   initialize other atom quantities for this sub-style
------------------------------------------------------------------------- */

int AtomVecThreeDipole::data_atom_hybrid(int nlocal, char **values)
{
  q[nlocal] = atof(values[0]);
  mu_x[nlocal][0] = atof(values[1]);
  mu_x[nlocal][1] = atof(values[2]);
  mu_x[nlocal][2] = atof(values[3]);
  mu_x[nlocal][3] = sqrt(mu_x[nlocal][0]*mu_x[nlocal][0] +
                       mu_x[nlocal][1]*mu_x[nlocal][1] +
                       mu_x[nlocal][2]*mu_x[nlocal][2]);
  mu_y[nlocal][0] = atof(values[4]);
  mu_y[nlocal][1] = atof(values[5]);
  mu_y[nlocal][2] = atof(values[6]);
  mu_y[nlocal][3] = sqrt(mu_y[nlocal][0]*mu_y[nlocal][0] +
                       mu_y[nlocal][1]*mu_y[nlocal][1] +
                       mu_y[nlocal][2]*mu_y[nlocal][2]);
  mu_z[nlocal][0] = atof(values[7]);
  mu_z[nlocal][1] = atof(values[8]);
  mu_z[nlocal][2] = atof(values[9]);
  mu_z[nlocal][3] = sqrt(mu_z[nlocal][0]*mu_z[nlocal][0] +
                        mu_z[nlocal][1]*mu_z[nlocal][1] +
                        mu_z[nlocal][2]*mu_z[nlocal][2]);
  return 10;
}

/* ----------------------------------------------------------------------
   pack atom info for data file including 3 image flags
------------------------------------------------------------------------- */

void AtomVecThreeDipole::pack_data(double **buf)
{
  int nlocal = atom->nlocal;
  for (int i = 0; i < nlocal; i++) {
    buf[i][0] = ubuf(tag[i]).d;
    buf[i][1] = ubuf(type[i]).d;
    buf[i][2] = q[i];
    buf[i][3] = x[i][0];
    buf[i][4] = x[i][1];
    buf[i][5] = x[i][2];
    buf[i][6] = mu_x[i][0];
    buf[i][7] = mu_x[i][1];
    buf[i][8] = mu_x[i][2];
    buf[i][9] = mu_y[i][0];
    buf[i][10] = mu_y[i][1];
    buf[i][11] = mu_y[i][2];
    buf[i][12] = mu_z[i][0];
    buf[i][13] = mu_z[i][1];
    buf[i][14] = mu_z[i][2];
    buf[i][15] = ubuf((image[i] & IMGMASK) - IMGMAX).d;
    buf[i][16] = ubuf((image[i] >> IMGBITS & IMGMASK) - IMGMAX).d;
    buf[i][17] = ubuf((image[i] >> IMG2BITS) - IMGMAX).d;
  }
}

/* ----------------------------------------------------------------------
   pack hybrid atom info for data file
------------------------------------------------------------------------- */

int AtomVecThreeDipole::pack_data_hybrid(int i, double *buf)
{
  buf[0] = q[i];
  buf[1] = mu_x[i][0];
  buf[2] = mu_x[i][1];
  buf[3] = mu_x[i][2];
  buf[4] = mu_y[i][0];
  buf[5] = mu_y[i][1];
  buf[6] = mu_y[i][2];
  buf[7] = mu_z[i][0];
  buf[8] = mu_z[i][1];
  buf[9] = mu_z[i][2];
  return 10;
}

/* ----------------------------------------------------------------------
   write atom info to data file including 3 image flags
------------------------------------------------------------------------- */

void AtomVecThreeDipole::write_data(FILE *fp, int n, double **buf)
{
  for (int i = 0; i < n; i++)
    fprintf(fp,TAGINT_FORMAT \
            " %d %-1.16e %-1.16e %-1.16e %-1.16e %-1.16e %-1.16e "
            "%-1.16e %-1.16e %-1.16e %-1.16e %-1.16e %-1.16e "
            "%-1.16e %d %d %d\n",
            (tagint) ubuf(buf[i][0]).i,(int) ubuf(buf[i][1]).i,
            buf[i][2],buf[i][3],buf[i][4],buf[i][5],
            buf[i][6],buf[i][7],buf[i][8],
            buf[i][9],buf[i][10],buf[i][11],
            buf[i][12],buf[i][13],buf[i][14],
            (int) ubuf(buf[i][15]).i,(int) ubuf(buf[i][16]).i,
            (int) ubuf(buf[i][17]).i);
}

/* ----------------------------------------------------------------------
   write hybrid atom info to data file
------------------------------------------------------------------------- */

int AtomVecThreeDipole::write_data_hybrid(FILE *fp, double *buf)
{
  fprintf(fp," %-1.16e %-1.16e %-1.16e %-1.16e "
             "%-1.16e %-1.16e %-1.16e "
             "%-1.16e %-1.16e %-1.16e\n",
             buf[0],buf[1],buf[2],buf[3],
             buf[4],buf[5],buf[6],
             buf[7],buf[8],buf[9]);
  return 10;
}

/* ----------------------------------------------------------------------
   return # of bytes of allocated memory
------------------------------------------------------------------------- */

bigint AtomVecThreeDipole::memory_usage()
{
  bigint bytes = 0;

  if (atom->memcheck("tag")) bytes += memory->usage(tag,nmax);
  if (atom->memcheck("type")) bytes += memory->usage(type,nmax);
  if (atom->memcheck("mask")) bytes += memory->usage(mask,nmax);
  if (atom->memcheck("image")) bytes += memory->usage(image,nmax);
  if (atom->memcheck("x")) bytes += memory->usage(x,nmax,3);
  if (atom->memcheck("v")) bytes += memory->usage(v,nmax,3);
  if (atom->memcheck("f")) bytes += memory->usage(f,nmax*comm->nthreads,3);

  if (atom->memcheck("q")) bytes += memory->usage(q,nmax);
  if (atom->memcheck("mu")) bytes += memory->usage(mu,nmax,4);
  if (atom->memcheck("mu_x")) bytes += memory->usage(mu,nmax,4);
  if (atom->memcheck("mu_y")) bytes += memory->usage(mu,nmax,4);
  if (atom->memcheck("mu_z")) bytes += memory->usage(mu,nmax,4);

  return bytes;
}
