#include "GLQuaternion4f.h"
#include <math.h>


		
GLQuaternion4f::GLQuaternion4f()
{
	imag =  new GLVector3f();
	real = 0;
}
GLQuaternion4f::GLQuaternion4f(GLfloat _real, GLfloat _imagI, GLfloat _imagJ, GLfloat _imagK)
{
	imag = new GLVector3f(_imagI, _imagJ, _imagK);
	real = _real;
}
GLQuaternion4f::GLQuaternion4f(GLfloat _real, GLVector3f* _imag)
{
	imag = new GLVector3f();
	real = _real;
}

GLQuaternion4f::~GLQuaternion4f(){}

GLvoid GLQuaternion4f::add(GLQuaternion4f* _quaternion)
{
	real += _quaternion->real;
	imag->add(_quaternion->imag);
}


GLvoid GLQuaternion4f::subtract(GLQuaternion4f* _quaternion)
{
	real -= _quaternion->real;
	imag->subtract(_quaternion->imag);
}

GLvoid GLQuaternion4f::multiply(GLQuaternion4f* _quaternion)
{
	GLfloat tx = imag->x;
	GLfloat ty = imag->y;
	GLfloat tz = imag->z;
	
	imag->x	= real * _quaternion->imag->x 	+ tx * _quaternion->real   		+ ty * _quaternion->imag->z		- tz * _quaternion->imag->y;
	imag->y	= real * _quaternion->imag->y 	- tx * _quaternion->imag->z 	+ ty * _quaternion->real		+ tz * _quaternion->imag->x;
	imag->z	= real * _quaternion->imag->z 	+ tx * _quaternion->imag->y 	- ty * _quaternion->imag->x 	+ tz * _quaternion->real;
	real	= real * _quaternion->real    	- tx * _quaternion->imag->x 	- ty * _quaternion->imag->y 	- tz * _quaternion->imag->z;
}


GLvoid GLQuaternion4f::polar(GLfloat _theta, GLVector3f* _vector)
{
	imag->x = _vector->x * (GLfloat) sin(_theta/2.0);
	imag->y = _vector->y * (GLfloat) sin(_theta/2.0);
	imag->z = _vector->z * (GLfloat) sin(_theta/2.0);
	real = (GLfloat) cos(_theta/2.0);
}

GLvoid GLQuaternion4f::polar(GLfloat _theta, GLfloat _x, GLfloat _y, GLfloat _z)
{
	imag->x = _x * (GLfloat) sin(_theta/2.0);
	imag->y = _y * (GLfloat) sin(_theta/2.0);
	imag->z = _z * (GLfloat) sin(_theta/2.0);
	real = (GLfloat) cos(_theta/2.0);
}

GLvoid GLQuaternion4f::reset()
{
	real = 1.0;
	imag->x = 0;
	imag->y = 0;
	imag->z = 0;
}

GLfloat GLQuaternion4f::length()
{
	return (GLfloat) sqrt(real * real + imag->x * imag->x + imag->y * imag->y + imag->z * imag->z);
}

GLvoid GLQuaternion4f::normalize()
{
	GLfloat l = length();
	if(l != 0.0)
	{
		real = real/l;
		imag->x = imag->x/l;
		imag->y = imag->y/l;
		imag->z = imag->z/l;	
	}
}

GLfloat* GLQuaternion4f::getRotationMatrix()
{
	GLfloat* m = (GLfloat*) malloc(16* sizeof(GLfloat));
	
	m[ 0] = 1.0 - 2.0 * imag->y * imag->y - 2.0 * imag->z * imag->z;
	m[ 1] = 2.0 * imag->x * imag->y + 2.0 * real * imag->z;
	m[ 2] = 2.0 * imag->x * imag->z - 2.0 * real * imag->y;
	m[ 3] = 0.0;
		
	m[ 4] = 2.0 * imag->x * imag->y - 2.0 * real * imag->z;
	m[ 5] = 1.0 - 2.0 * imag->x * imag->x - 2.0 * imag->z * imag->z;
	m[ 6] = 2.0 * imag->y * imag->z + 2.0 * real * imag->x;
	m[ 7] = 0.0;
		
	m[ 8] = 2.0 * imag->x * imag->z + 2.0 * real * imag->y;
	m[ 9] = 2.0 * imag->y * imag->z - 2.0 * real * imag->x;
	m[10] = 1.0 - 2.0 * imag->x * imag->x - 2.0 * imag->y * imag->y;
	m[11] = 0.0;
		
	m[12] = 0.0;
	m[13] = 0.0;
	m[14] = 0.0;
	m[15] = 1.0;

	return m;
}
	
