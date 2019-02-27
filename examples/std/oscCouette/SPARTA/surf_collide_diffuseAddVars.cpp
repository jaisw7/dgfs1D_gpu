/* ----------------------------------------------------------------------
   SPARTA - Stochastic PArallel Rarefied-gas Time-accurate Analyzer
   http://sparta.sandia.gov
   Steve Plimpton, sjplimp@sandia.gov, Michael Gallis, magalli@sandia.gov
   Sandia National Laboratories

   Copyright (2014) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under 
   the GNU General Public License.

   See the README file in the top-level SPARTA directory.
------------------------------------------------------------------------- */

#include "math.h"
#include "stdlib.h"
#include "string.h"
#include "surf_collide_diffuseAddVars.h"
#include "surf.h"
#include "surf_react.h"
#include "input.h"
#include "variable.h"
#include "particle.h"
#include "domain.h"
#include "update.h"
#include "modify.h"
#include "comm.h"
#include "random_mars.h"
#include "random_park.h"
#include "math_const.h"
#include "math_extra.h"
#include "error.h"

using namespace SPARTA_NS;
using namespace MathConst;

/* ---------------------------------------------------------------------- */

SurfCollideDiffuseAddVars::SurfCollideDiffuseAddVars(SPARTA *sparta, int narg, char **arg) :
  SurfCollide(sparta, narg, arg)
{
  if (narg < 4) error->all(FLERR,"Illegal surf_collide diffuseAddVars command");

  tstr = NULL;
  vxstr = NULL;
  vystr = NULL;
  vzstr = NULL;
  

  if (strstr(arg[2],"v_") == arg[2]) {
    int n = strlen(&arg[2][2]) + 1;
    tstr = new char[n];
    strcpy(tstr,&arg[2][2]);
  } else {
    twall = input->numeric(FLERR,arg[2]); 
    if (twall <= 0.0) error->all(FLERR,"Surf_collide diffuseAddVars temp <= 0.0");
  }

  acc = input->numeric(FLERR,arg[3]); 
  if (acc < 0.0 || acc > 1.0) 
    error->all(FLERR,"Illegal surf_collide diffuseAddVars command");

  // optional args

  tflag = rflag = 0;

  int iarg = 4;
  while (iarg < narg) {
    if (strcmp(arg[iarg],"translate") == 0) {
      if (iarg+4 > narg) 
        error->all(FLERR,"Illegal surf_collide diffuseAddVars command");
      tflag = 1;
      //vx = atof(arg[iarg+1]);
      //vy = atof(arg[iarg+2]);
      //vz = atof(arg[iarg+3]);

    if (strstr(arg[iarg+1],"v_") == arg[iarg+1]) {
        int n = strlen(&arg[iarg+1][2]) + 1;
        vxstr = new char[n];
        strcpy(vxstr,&arg[iarg+1][2]);
      } else {
        vx = input->numeric(FLERR,arg[iarg+1]); 
      }

    if (strstr(arg[iarg+2],"v_") == arg[iarg+2]) {
        int n = strlen(&arg[iarg+2][2]) + 1;
        vystr = new char[n];
        strcpy(vystr,&arg[iarg+2][2]);
      } else {
        vy = input->numeric(FLERR,arg[iarg+2]); 
      }

    if (strstr(arg[iarg+3],"v_") == arg[iarg+3]) {
        int n = strlen(&arg[iarg+3][2]) + 1;
        vzstr = new char[n];
        strcpy(vzstr,&arg[iarg+3][2]);
      } else {
        vz = input->numeric(FLERR,arg[iarg+3]); 
      }

      //printf("%lf, %lf, %lf ", vx, vy, vz);

      iarg += 4;
    } else if (strcmp(arg[iarg],"rotate") == 0) {
      if (iarg+7 > narg) 
        error->all(FLERR,"Illegal surf_collide diffuseAddVars command");
      rflag = 1;
      px = atof(arg[iarg+1]);
      py = atof(arg[iarg+2]);
      pz = atof(arg[iarg+3]);
      wx = atof(arg[iarg+4]);
      wy = atof(arg[iarg+5]);
      wz = atof(arg[iarg+6]);
      if (domain->dimension == 2 && pz != 0.0) 
        error->all(FLERR,"Surf_collide diffuseAddVars rotation invalid for 2d");
      if (domain->dimension == 2 && (wx != 0.0 || wy != 0.0))
        error->all(FLERR,"Surf_collide diffuseAddVars rotation invalid for 2d");
      iarg += 7;
    } else error->all(FLERR,"Illegal surf_collide diffuseAddVars command");
  }

  if (tflag && rflag) error->all(FLERR,"Illegal surf_collide diffuseAddVars command");
  if (tflag || rflag) trflag = 1;
  else trflag = 0;

  // initialize RNG

  random = new RanPark(update->ranmaster->uniform());
  double seed = update->ranmaster->uniform();
  random->reset(seed,comm->me,100);

  //printf("%lf, %lf, %lf ", vx, vy, vz);
}

/* ---------------------------------------------------------------------- */

SurfCollideDiffuseAddVars::~SurfCollideDiffuseAddVars()
{
  delete [] tstr;
  delete random;
  delete [] vxstr;
  delete [] vystr;
  delete [] vzstr;
}

/* ---------------------------------------------------------------------- */

void SurfCollideDiffuseAddVars::init()
{
  SurfCollide::init();

  // check variable

  if (tstr) {
    tvar = input->variable->find(tstr);
    if (tvar < 0) 
      error->all(FLERR,"Surf_collide diffuseAddVars variable name does not exist");
    if (!input->variable->equal_style(tvar))
      error->all(FLERR,"Surf_collide diffuseAddVars variable is invalid style");
    
    twall = input->variable->compute_equal(tvar);
      if (twall <= 0.0) error->all(FLERR,"Surf_collide diffuseAddVars temp <= 0.0");
  }

  if (vxstr) {
    vxvar = input->variable->find(vxstr);
    if (vxvar < 0) 
      error->all(FLERR,"Surf_collide diffuseAddVars variable name does not exist");
    if (!input->variable->equal_style(vxvar))
      error->all(FLERR,"Surf_collide diffuseAddVars variable is invalid style");

    vx = input->variable->compute_equal(vxvar);
  }

  if (vystr) {
    vyvar = input->variable->find(vystr);
    if (vyvar < 0) 
      error->all(FLERR,"Surf_collide diffuseAddVars variable name does not exist");
    if (!input->variable->equal_style(vyvar))
      error->all(FLERR,"Surf_collide diffuseAddVars variable is invalid style");

    vy = input->variable->compute_equal(vyvar);
  }

  if (vzstr) {
    vzvar = input->variable->find(vzstr);
    if (vzvar < 0) 
      error->all(FLERR,"Surf_collide diffuseAddVars variable name does not exist");
    if (!input->variable->equal_style(vzvar))
      error->all(FLERR,"Surf_collide diffuseAddVars variable is invalid style");

    vz = input->variable->compute_equal(vzvar);
  }

}

/* ----------------------------------------------------------------------
   particle collision with surface with optional chemistry
   ip = particle with current x = collision pt, current v = incident v
   norm = surface normal unit vector
   isr = index of reaction model if >= 0, -1 for no chemistry
   ip = set to NULL if destroyed by chemsitry
   return jp = new particle if created by chemistry
   resets particle(s) to post-collision outward velocity
------------------------------------------------------------------------- */

Particle::OnePart *SurfCollideDiffuseAddVars::
collide(Particle::OnePart *&ip, double *norm, double &dtremain, int isr)
{
  //this->dynamic();
  nsingle++;

  // if surface chemistry defined, attempt reaction
  // reaction = 1 if reaction took place

  Particle::OnePart iorig;
  Particle::OnePart *jp = NULL;
  int reaction = 0;

  if (isr >= 0) {
    if (modify->n_surf_react) memcpy(&iorig,ip,sizeof(Particle::OnePart));
    reaction = surf->sr[isr]->react(ip,norm,jp);
    if (reaction) surf->nreact_one++;
  }

  // diffuseAddVars reflection for each particle

  if (ip) diffuseAddVars(ip,norm);
  if (jp) diffuseAddVars(jp,norm);

  // call any fixes with a surf_react() method
  // they may reset j to -1, e.g. fix ambipolar
  //   in which case newly created j is deleted

  if (reaction && modify->n_surf_react) {
    int i = -1;
    if (ip) i = ip - particle->particles;
    int j = -1;
    if (jp) j = jp - particle->particles;
    modify->surf_react(&iorig,i,j);
    if (jp && j < 0) {
      jp = NULL;
      particle->nlocal--;
    }
  }

  return jp;
}

/* ----------------------------------------------------------------------
   particle collision with surface with optional chemistry
   ip = particle with current x = collision pt, current v = incident v
   norm = surface normal unit vector
   ip = set to NULL if destroyed by chemsitry
   return jp = new particle if created by chemistry
   resets particle(s) to post-collision outward velocity
------------------------------------------------------------------------- */

void SurfCollideDiffuseAddVars::diffuseAddVars(Particle::OnePart *p, double *norm)
{
  // specular reflection
  // reflect incident v around norm

  //if(p->id==1)
  //  printf("-------------- %lf, %lf, %lf, %lf\n", vx, vy, vz, twall);

  if (random->uniform() > acc) {
    MathExtra::reflect3(p->v,norm);

  // diffuseAddVars reflection
  // vrm = most probable speed of species, eqns (4.1) and (4.7)
  // vperp = velocity component perpendicular to surface along norm, eqn (12.3)
  // vtan12 = 2 velocity components tangential to surface
  // tangent1 = component of particle v tangential to surface,
  //   check if tangent1 = 0 (normal collision), set randomly
  // tangent2 = norm x tangent1 = orthogonal tangential direction
  // tangent12 are both unit vectors

  } else {
    this->dynamic();
    //if(tvar)
    //  twall = input->variable->compute_equal(tvar);
    //if (twall <= 0.0) error->all(FLERR,"Surf_collide diffuseAddVars temp <= 0.0");

    //vx = input->variable->compute_equal(vxvar);
    //vy = input->variable->compute_equal(vyvar);
    //vz = input->variable->compute_equal(vzvar);

    double tangent1[3],tangent2[3];
    Particle::Species *species = particle->species;
    int ispecies = p->ispecies;

    double vrm = sqrt(2.0*update->boltz * twall / species[ispecies].mass);
    double vperp = vrm * sqrt(-log(random->uniform()));

    double theta = MY_2PI * random->uniform();
    double vtangent = vrm * sqrt(-log(random->uniform()));
    double vtan1 = vtangent * sin(theta);
    double vtan2 = vtangent * cos(theta);

    double *v = p->v;
    double dot = MathExtra::dot3(v,norm);

    double beta_un,normalized_distbn_fn;

    tangent1[0] = v[0] - dot*norm[0];
    tangent1[1] = v[1] - dot*norm[1];
    tangent1[2] = v[2] - dot*norm[2];

    if (MathExtra::lensq3(tangent1) == 0.0) {
      tangent2[0] = random->uniform();
      tangent2[1] = random->uniform();
      tangent2[2] = random->uniform();
      MathExtra::cross3(norm,tangent2,tangent1);
    }

    MathExtra::norm3(tangent1);
    MathExtra::cross3(norm,tangent1,tangent2);

    // add in translation or rotation vector if specified
    // only keep portion of vector tangential to surface element

    if (trflag) {
      double vxdelta,vydelta,vzdelta;
      if (tflag) {
        vxdelta = vx; vydelta = vy; vzdelta = vz;
        double dot = vxdelta*norm[0] + vydelta*norm[1] + vzdelta*norm[2];
     
        if (fabs(dot) > 0.001) {
          dot /= vrm;
          do {
            do {
              beta_un = (6.0*random->uniform() - 3.0);
            } while (beta_un + dot < 0.0);
            normalized_distbn_fn = 2.0 * (beta_un + dot) /
              (dot + sqrt(dot*dot + 2.0)) *
              exp(0.5 + (0.5*dot)*(dot-sqrt(dot*dot + 2.0)) -
                  beta_un*beta_un);
          } while (normalized_distbn_fn < random->uniform());
          vperp = beta_un*vrm;
        }

      } else {
        double *x = p->x;
        vxdelta = wy*(x[2]-pz) - wz*(x[1]-py);
        vydelta = wz*(x[0]-px) - wx*(x[2]-pz);
        vzdelta = wx*(x[1]-py) - wy*(x[0]-px);
        double dot = vxdelta*norm[0] + vydelta*norm[1] + vzdelta*norm[2];
        vxdelta -= dot*norm[0];
        vydelta -= dot*norm[1];
        vzdelta -= dot*norm[2];
      }
      
      v[0] = vperp*norm[0] + vtan1*tangent1[0] + vtan2*tangent2[0] + vxdelta;
      v[1] = vperp*norm[1] + vtan1*tangent1[1] + vtan2*tangent2[1] + vydelta;
      v[2] = vperp*norm[2] + vtan1*tangent1[2] + vtan2*tangent2[2] + vzdelta;

    // no translation or rotation

    } else {
      v[0] = vperp*norm[0] + vtan1*tangent1[0] + vtan2*tangent2[0];
      v[1] = vperp*norm[1] + vtan1*tangent1[1] + vtan2*tangent2[1];
      v[2] = vperp*norm[2] + vtan1*tangent1[2] + vtan2*tangent2[2];
    }

    p->erot = particle->erot(ispecies,twall,random);
    p->evib = particle->evib(ispecies,twall,random);
  }
}

/* ----------------------------------------------------------------------
   set current surface temperature
------------------------------------------------------------------------- */

void SurfCollideDiffuseAddVars::dynamic()
{
  if(tstr)
  {
    twall = input->variable->compute_equal(tvar);
    if (twall <= 0.0) error->all(FLERR,"Surf_collide diffuseAddVars temp <= 0.0");
  }

  if(vxstr)
    vx = input->variable->compute_equal(vxvar);
  
  if(vystr)
    vy = input->variable->compute_equal(vyvar);

  if(vzstr)
    vz = input->variable->compute_equal(vzvar);

  //printf("-------------- %lf, %lf, %lf ", vx, vy, vz);
  //error->all(FLERR,"Surf_collide diffuseAddVars temp <= 0.0");
}
