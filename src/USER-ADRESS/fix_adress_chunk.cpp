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

#include "fix_adress_chunk.h"
#include <cmath>
#include <cstring>
#include "atom.h"
#include "update.h"
#include "force.h"
#include "respa.h"
#include "modify.h"
#include "compute_chunk_atom.h"
#include "compute_com_chunk.h"
#include "memory.h"
#include "error.h"
#include "domain.h"
#include "math.h"
#include "math_const.h"

using namespace LAMMPS_NS;
using namespace FixConst;
using namespace MathConst;

#define SMALL 1.0e-10

/* ---------------------------------------------------------------------- */

FixAdressChunk::FixAdressChunk(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg),
  idchunk(NULL), idcom(NULL), com0(NULL), fcom(NULL)
{
  if (atom->adress_flag != 1) error->all(FLERR,"fix adress/chunk requires the adress atom style");
  if (narg != 7) error->all(FLERR,"Illegal fix adress/chunk command");

  scalar_flag = 1;
  global_freq = 1;
  extscalar = 1;
  respa_level_support = 1;
  ilevel_respa = 0;

  r_explicit = force->numeric(FLERR,arg[3]);
  r_coarsegr = force->numeric(FLERR,arg[4]);

  int n = strlen(arg[5]) + 1;
  idchunk = new char[n];
  strcpy(idchunk,arg[5]);

  n = strlen(arg[6]) + 1;
  idcom = new char[n];
  strcpy(idcom,arg[6]);

  nchunk = 0;
}

/* ---------------------------------------------------------------------- */

FixAdressChunk::~FixAdressChunk()
{
  memory->destroy(com0);
  memory->destroy(chunkw);
  memory->destroy(fcom);

  // decrement lock counter in compute chunk/atom, it if still exists

  int icompute = modify->find_compute(idchunk);
  if (icompute >= 0) {
    cchunk = (ComputeChunkAtom *) modify->compute[icompute];
    cchunk->unlock(this);
    cchunk->lockcount--;
  }

  delete [] idchunk;
  delete [] idcom;
}

/* ---------------------------------------------------------------------- */

int FixAdressChunk::setmask()
{
  int mask = 0;
  // mask |= POST_FORCE;
  // mask |= POST_FORCE_RESPA;
  mask |= POST_INTEGRATE;
  mask |= POST_INTEGRATE_RESPA;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixAdressChunk::init()
{
  // current indices for idchunk and idcom

  int icompute = modify->find_compute(idchunk);
  if (icompute < 0)
    error->all(FLERR,"Chunk/atom compute does not exist for fix adress/chunk");
  cchunk = (ComputeChunkAtom *) modify->compute[icompute];
  if (strcmp(cchunk->style,"chunk/atom") != 0)
    error->all(FLERR,"Fix adress/chunk does not use chunk/atom compute");

  icompute = modify->find_compute(idcom);
  if (icompute < 0)
    error->all(FLERR,"Com/chunk compute does not exist for fix adress/chunk");
  ccom = (ComputeCOMChunk *) modify->compute[icompute];
  if (strcmp(ccom->style,"com/chunk") != 0)
    error->all(FLERR,"Fix adress/chunk does not use com/chunk compute");

  // check that idchunk is consistent with ccom->idchunk

  if (strcmp(idchunk,ccom->idchunk) != 0)
    error->all(FLERR,"Fix spring chunk chunkID not same as comID chunkID");

  if (strstr(update->integrate_style,"respa")) {
    ilevel_respa = ((Respa *) update->integrate)->nlevels-1;
    if (respa_level >= 0) ilevel_respa = MIN(respa_level,ilevel_respa);
  }
}

/* ---------------------------------------------------------------------- */

void FixAdressChunk::setup(int vflag)
{
//  if (strstr(update->integrate_style,"verlet"))
//    post_force(vflag);
//  else {
//    ((Respa *) update->integrate)->copy_flevel_f(ilevel_respa);
//    post_force_respa(vflag,ilevel_respa,0);
//    ((Respa *) update->integrate)->copy_f_flevel(ilevel_respa);
//  }
}

/* ---------------------------------------------------------------------- */

void FixAdressChunk::post_integrate()
{
  int i,m;
  double dx,dy,dz,r;

  // calculate current centers of mass for each chunk
  // extract pointers from idchunk and idcom

  ccom->compute_array();

  nchunk = cchunk->nchunk;
  int *ichunk = cchunk->ichunk;
  double *masstotal = ccom->masstotal;
  double **com = ccom->array;

  if (com0 == NULL) cchunk->lock(this,update->ntimestep,-1);

  if (com0 == NULL) {
    memory->create(com0,3,"adress/chunk:com0");
    memory->create(chunkw,nchunk,"adress/chunk:chunkw");
    memory->create(fcom,nchunk,3,"adress/chunk:fcom");
  }

  com0[0] = (domain->boxlo[0]+domain->boxhi[0])/2;
  com0[1] = (domain->boxlo[1]+domain->boxhi[1])/2;
  com0[2] = (domain->boxlo[2]+domain->boxhi[2])/2;
    
  // calculate the AdResS weight for each chunk

  int nlocal = atom->nlocal;
  double *w = atom->adw;

  for (m = 0; m < nchunk; m++) {
    dx = com[m][0] - com0[0];
    dy = com[m][1] - com0[1];
    dz = com[m][2] - com0[2];
    r = sqrt(dx*dx + dy*dy + dz*dz);
    if (r < r_explicit) {
      chunkw[m] = 1.0;
    } else if (r > r_coarsegr) {
      chunkw[m] = 0.0;
    } else {
      chunkw[m] = cos(MY_PI/(2*r_coarsegr)*(r-r_explicit));
      chunkw[m] *= chunkw[m];
    }
  }

  // inform the chunklets

  for (i = 0; i < nlocal; i++) {
    m = ichunk[i]-1;
    if (m < 0) continue;
    w[i] = chunkw[m];
  }

}

/* ---------------------------------------------------------------------- */
void FixAdressChunk::post_integrate_respa(int ilevel, int /*iloop*/)
{
  if (ilevel == ilevel_respa) post_integrate();
}

