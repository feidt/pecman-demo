#include "GLVector3f.h"
#include <math.h>

GLVector3f::GLVector3f()
{
	x = 0.0;
	y = 0.0;
	z = 0.0;
}
GLVector3f::GLVector3f(GLfloat _x, GLfloat _y, GLfloat _z)
{
	x = _x;
	y = _y;
	z = _z;
}
GLVector3f::~GLVector3f(){}

GLvoid GLVector3f::add(GLVector3f* _vector)
{
	x += _vector->x;
	y += _vector->y;
	z += _vector->z;
}
GLvoid GLVector3f::subtract(GLVector3f* _vector)
{
	x -= _vector->x;
	y -= _vector->y;
	z -= _vector->z;
}

GLfloat GLVector3f::dotProduct(GLVector3f* _vector)
{
	return (GLfloat) ( x*_vector->x + y * _vector->y + z * _vector->z);
}

GLvoid GLVector3f::crossProduct(GLVector3f* _vector)
{
	GLfloat tx = x, ty = y ,tz = z;
	x = ty * _vector->z - tz * _vector->y;
	y = tz * _vector->x - tx * _vector->z;
	z = tx * _vector->y - ty * _vector->x;
}

GLvoid GLVector3f::set(GLVector3f* _vector)
{
	x = _vector->x;
	y = _vector->y;
	z = _vector->z;
}

GLfloat GLVector3f::length()
{
	return (float) sqrt(x*x + y*y + z*z);
}
GLvoid GLVector3f::normalize()
{
	float l = length();
	if(l != 0.0)
	{
		x = x/l;
		y = y/l;
		z = z/l;
	}
}
