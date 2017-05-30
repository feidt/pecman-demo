#include <GL/glut.h>
#include "GLVector3f.h"

class GLQuaternion4f
{
	public:
		GLfloat real;
		GLVector3f* imag;
		GLQuaternion4f();
		GLQuaternion4f(GLfloat _real, GLfloat _imagI, GLfloat _imagJ, GLfloat _imagK);
		GLQuaternion4f(GLfloat _real, GLVector3f* _imag);
		~GLQuaternion4f();

		GLvoid add(GLQuaternion4f* _quaternion);
		GLvoid subtract(GLQuaternion4f* _quaternion);
		GLvoid multiply(GLQuaternion4f* _quaternion);
		GLvoid polar(GLfloat _theta, GLVector3f* _vector);
		GLvoid polar(GLfloat _theta, GLfloat _x, GLfloat _y, GLfloat _z);
		GLvoid reset();
		GLfloat length();
		GLvoid normalize();
		GLfloat* getRotationMatrix();
		
		 
};
