// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#include "hoomd/HOOMDMath.h"
#include "hoomd/VectorMath.h"
#include "hoomd/RandomNumbers.h"
#include "hoomd/Hypersphere.h"

/*! \file Moves.h
    \brief Trial move generators
*/

#ifndef __MOVES_H__
#define __MOVES_H__

// need to declare these class methods with __device__ qualifiers when building in nvcc
// DEVICE is __host__ __device__ when included in nvcc and blank when included into the host compiler
#ifdef NVCC
#define DEVICE __device__
#else
#define DEVICE
#endif

//! hpmc namespace
namespace hpmc
{

//! Translation move
/*! \param v Vector to translate (in/out)
    \param rng random123 RNG to utilize in the move
    \param d Maximum move distance
    \param dim Dimension

    When \a dim == 2, only x and y components are moved.
*/
template <class RNG>
DEVICE inline void move_translate(vec3<Scalar>& v, RNG& rng, Scalar d, unsigned int dim)
    {
    hoomd::UniformDistribution<Scalar> uniform(-d, d);

    // Generate a random vector inside a sphere of radius d
    vec3<Scalar> dr(Scalar(0.0), Scalar(0.0), Scalar(0.0));
    do
        {
        dr.x = uniform(rng);
        dr.y = uniform(rng);
        if (dim != 2)
            dr.z = uniform(rng);
        } while(dot(dr,dr) > d*d);

    // apply the move vector
    v += dr;
    }

//! Pure translation move on the hypersphere along a geodesic
/*! \param quat_l Left quaternion of sphere coordinate to translate (in/out)
    \param quat_r Right quaternion of sphere coordinate to translate (in/out)
    \param rng Saru RNG to utilize in the move
    \param d Maximum arc-length (in distance units)
    \param hypersphere The hypersphere boundary

    See Sinkovits et al JCP 2011 for a definition.
*/
template <class RNG>
DEVICE inline void move_translate_hypersphere(quat<Scalar>& quat_l, quat<Scalar>& quat_r, RNG& rng, Scalar d, const Hypersphere& hypersphere)
    {
    // Generate a random arc of maximum length d

    hoomd::UniformDistribution<Scalar> uniform(Scalar(-1.0),Scalar(1.0));

    Scalar phi = d*uniform(rng)/hypersphere.getR();

    vec3<Scalar> b;

    //! Generate a direction (3d unit vector) for the translation
    Scalar theta = hoomd::UniformDistribution<Scalar>(Scalar(0.0),Scalar(2.0*M_PI))(rng);
    Scalar z = uniform(rng);
    b = vec3<Scalar>(fast::sqrt(Scalar(1.0)-z*z)*fast::cos(theta),fast::sqrt(Scalar(1.0)-z*z)*fast::sin(theta),z);

    // the transformation quaternion
    quat<Scalar> p(fast::cos(0.5*phi),fast::sin(0.5*phi)*b);

    // apply the translation in the standard position
    quat_l = quat_l*p;

    // renormalize
    Scalar norm_l_inv = fast::rsqrt(dot(quat_l, quat_l));
    quat_l.s *= norm_l_inv;
    quat_l.v *= norm_l_inv;

    quat_r = p*quat_r;

    Scalar norm_r_inv = fast::rsqrt(dot(quat_r, quat_r));
    quat_r.s *= norm_r_inv;
    quat_r.v *= norm_r_inv;
    }

//! Rotation move
/*! \param orientation Quaternion to rotate (in/out)
    \param rng random123 RNG to utilize in the move
    \param a Rotation magnitude
    \param dim Dimension

    When \a dim == 2, a random rotation about (0,0,1) is generated. When \a dim == 3 a random 3D rotation is generated.
*/
template <class RNG>
DEVICE void move_rotate(quat<Scalar>& orientation, RNG& rng, Scalar a, unsigned int dim)
    {
    if (dim==2)
        {
        a /= Scalar(2.0);
        Scalar alpha = hoomd::UniformDistribution<Scalar>(-a, a)(rng);;
        quat<Scalar> q(fast::cos(alpha), fast::sin(alpha) * vec3<Scalar>(Scalar(0),Scalar(0),Scalar(1))); // rotation quaternion
        orientation = orientation * q;
        orientation = orientation * (fast::rsqrt(norm2(orientation)));
        }
    else
        {
        hoomd::UniformDistribution<Scalar> uniform(Scalar(-1.0), Scalar(1.0));

        // Frenkel and Smit reference Allen and Tildesley, referencing Vesley(1982), referencing Marsaglia(1972).
        // Generate a random unit quaternion. Scale it to a small rotation and apply.
        quat<Scalar> q;
        Scalar s1, s2, s3;

        do
            {
            q.s = uniform(rng);
            q.v.x = uniform(rng);
            }
        while ((s1 = q.s * q.s + q.v.x * q.v.x) >= Scalar(1.0));

        do
            {
            q.v.y = uniform(rng);
            q.v.z = uniform(rng);
            }
        while ((s2 = q.v.y * q.v.y + q.v.z * q.v.z) >= Scalar(1.0) || s2 == Scalar(0.0));

        s3 = fast::sqrt((Scalar(1.0) - s1) / s2);
        q.v.y *= s3;
        q.v.z *= s3;

        // generate new trial orientation
        orientation += a * q;

        // renormalize
        orientation = orientation * (fast::rsqrt(norm2(orientation)));
        }
    }

//! Pure rotation move on the hypersphere around a randomly chosen axis
/*! \param quat_l Left quaternion of sphere coordinate to translate (in/out)
    \param quat_r Right quaternion of sphere coordinate to translate (in/out)
    \param rng Saru RNG to utilize in the move
    \param a Maximum rotation angle, in radians

    See Sinkovits et al JCP 2011 for a definition.

*/
template <class RNG>
DEVICE inline void move_rotate_hypersphere(quat<Scalar>& quat_l, quat<Scalar>& quat_r, RNG& rng, Scalar a)
    {
    // Generate a random angle between -a/2 and a/2

    hoomd::UniformDistribution<Scalar> uniform(Scalar(-1.0),Scalar(1.0));

    Scalar phi = a*uniform(rng);

    vec3<Scalar> b;

    //! Generate a direction (3d unit vector) for the translation
    Scalar theta = hoomd::UniformDistribution<Scalar>(Scalar(0.0),Scalar(2.0*M_PI))(rng);
    Scalar z = uniform(rng);
    b = vec3<Scalar>(fast::sqrt(Scalar(1.0)-z*z)*fast::cos(theta),fast::sqrt(Scalar(1.0)-z*z)*fast::sin(theta),z);

    // the transformation quaternion
    quat<Scalar> p(fast::cos(0.5*phi),fast::sin(0.5*phi)*b);

    // apply the rotation in the standard position
    quat_l = quat_l*p;

    quat_r = conj(p)*quat_r;

    // renormalize
    Scalar norm_l_inv = fast::rsqrt(dot(quat_l, quat_l));
    quat_l.s *= norm_l_inv;
    quat_l.v *= norm_l_inv;

    Scalar norm_r_inv = fast::rsqrt(dot(quat_r, quat_r));
    quat_r.s *= norm_r_inv;
    quat_r.v *= norm_r_inv;
    }

//! Select a random index
/*! \param rng Saru RNG to utilize in the move
    \param max Maximum index to select
    \returns a random number 0 <= i <= max with uniform probability.

    **Method**

    First, round max+1 up to the next nearest power of two -> max2. Then draw random numbers in the range [0 ... max2)
    using 32-but random values and a bitwise and with max2-1. Return the first random number found in the range.
*/
template <class RNG>
DEVICE inline unsigned int rand_select(RNG& rng, unsigned int max)
    {
    // handle degenerate case where max==0
    if (max == 0)
        return 0;

    // algorithm to round up to the nearest power of two from https://en.wikipedia.org/wiki/Power_of_two
    unsigned int n = max+1;
    n = n - 1;
    n = n | (n >> 1);
    n = n | (n >> 2);
    n = n | (n >> 4);
    n = n | (n >> 8);
    n = n | (n >> 16);
    // Note: leaving off the n = n + 1 because we are going to & with next highest power of 2 -1

    unsigned int result;
    do
        {
        result = rng.u32() & n;
        } while(result > max);

    return result;
    }


//! Helper function to test if a particle is in an active region
/*! \param pos Position of the particle
    \param box simulation box
    \param ghost_fraction Fraction of the box in the inactive zone
*/
DEVICE inline bool isActive(Scalar3 pos, const BoxDim& box, Scalar3 ghost_fraction)
    {
    // Determine if particle is in the active region
    Scalar3 f = box.makeFraction(pos);
    uchar3 periodic = box.getPeriodic();
    if ( (!periodic.x && (f.x < Scalar(0.0) || f.x >= (Scalar(1.0) - ghost_fraction.x))) ||
         (!periodic.y && (f.y < Scalar(0.0) || f.y >= (Scalar(1.0) - ghost_fraction.y))) ||
         (!periodic.z && (f.z < Scalar(0.0) || f.z >= (Scalar(1.0) - ghost_fraction.z))) )
        {
        return false;
        }
    return true;
    }

//! Helper function to generate a random element of SO(3)
// see Shoemake, Uniform random rotations, Graphics Gems III, p.142-132
// and http://math.stackexchange.com/questions/131336/uniform-random-quaternion-in-a-restricted-angle-range
template<class RNG>
DEVICE inline quat<Scalar> generateRandomOrientation(RNG& rng)
    {
    Scalar u1 = hoomd::detail::generate_canonical<Scalar>(rng);
    Scalar u2 = hoomd::detail::generate_canonical<Scalar>(rng);
    Scalar u3 = hoomd::detail::generate_canonical<Scalar>(rng);
    return quat<Scalar>(fast::sqrt(u1)*fast::cos(Scalar(2.0*M_PI)*u3),
        vec3<Scalar>(fast::sqrt(Scalar(1.0)-u1)*fast::sin(Scalar(2.0*M_PI)*u2),
            fast::sqrt(Scalar(1.0-u1))*fast::cos(Scalar(2.0*M_PI)*u2),
            fast::sqrt(u1)*fast::sin(Scalar(2.0*M_PI)*u3)));

    }

//! Generate a random rotation about the z-axis
template<class RNG>
DEVICE inline quat<Scalar> generateRandomOrientation2D(RNG& rng)
    {
    Scalar theta = hoomd::UniformDistribution<Scalar>(-M_PI, M_PI)(rng);
    return quat<Scalar>(fast::cos(theta/2.0), vec3<Scalar>(0, 0, fast::sin(theta/2.0)));
    }

/* Generate a uniformly distributed random position in a sphere
 * \param rng random123 RNG to use to generate the position
 * \param pos_sphere Center of insertion sphere
 * \param R radius of insertion sphere
 */
template<class RNG>
inline vec3<Scalar> generatePositionInSphere(RNG& rng, vec3<Scalar> pos_sphere, Scalar R)
    {
    // random normalized vector
    vec3<Scalar> n;
    hoomd::SpherePointGenerator<Scalar>()(rng, n);

    // draw random radial coordinate in test sphere
    Scalar r3 = hoomd::detail::generate_canonical<Scalar>(rng);
    Scalar r = R*fast::pow(r3,Scalar(1.0/3.0));

    // test depletant position
    vec3<Scalar> pos_in_sphere = pos_sphere+r*n;

    return pos_in_sphere;
    }

/* Generate a uniformly distributed random position in a spherical cap
 *
 * \param rng The random number generator
 * \param pos_sphere Center of sphere
 * \param R radius of sphere
 * \param h height of spherical cap (0<h<2*R)
 * \param d Vector normal to the cap
 */
template<class RNG>
inline vec3<Scalar> generatePositionInSphericalCap(RNG& rng, const vec3<Scalar>& pos_sphere,
     Scalar R, Scalar h, const vec3<Scalar>& d)
    {
    // pick a z coordinate in the spherical cap s.t. V(z) ~ uniform
    Scalar theta = Scalar(2.0*M_PI)*hoomd::detail::generate_canonical<Scalar>(rng);
    Scalar R3=R*R*R;
    Scalar V_cap = Scalar(M_PI/3.0)*h*h*(Scalar(3.0)*R-h);
    Scalar V = V_cap*hoomd::detail::generate_canonical<Scalar>(rng);
    const Scalar sqrt3(1.7320508075688772935);

    // convert the cap volume into a z coordinate in the sphere, using the correct root of the cubic polynomial
    Scalar arg = Scalar(1./3.)*atan2(fast::sqrt((Scalar(4.0*M_PI)*R3-Scalar(3.0)*V)*3*V),Scalar(2.0*M_PI)*R3-Scalar(3.0)*V);
    Scalar z = R*(fast::cos(arg)-sqrt3*fast::sin(arg));

    // pick a point in disk of radius sqrt(R^2-z^2)
    Scalar r = fast::sqrt(rng.template s<Scalar>()*(R*R-z*z));

    // unit vector in cap direction
    vec3<Scalar> n = d/sqrt(dot(d,d));

    // find two unit vectors normal to n
    vec3<Scalar> ez(0,0,1);
    vec3<Scalar> n1, n2;
    vec3<Scalar> c = cross(n,ez);
    if (dot(c,c)==0.0)
        {
        n1 = vec3<Scalar>(1,0,0);
        n2 = vec3<Scalar>(0,1,0);
        }
    else
        {
        n1 = c/sqrt(dot(c,c));
        c = cross(n,n1);
        n2 = c/sqrt(dot(c,c));
        }

    vec3<Scalar> r_cone = n1*r*cos(theta)+n2*r*sin(theta)+n*z;

    // test depletant position
    return pos_sphere+r_cone;
    }

/*! Reflect a point in R3 around a line (pi rotation), given by a point p through which it passes
    and a rotation quaternion
 */
inline vec3<Scalar> lineReflection(vec3<Scalar> pos, vec3<Scalar> p, quat<Scalar> q)
    {
    // find closest point on line
    vec3<Scalar> n = q.v;
    Scalar t = dot(pos-p,n);
    vec3<Scalar> r = p + t*n;

    // pivot around that point
    return r - (pos - r);
    }

}; // end namespace hpmc

#undef DEVICE

#endif //__MOVES_H__
