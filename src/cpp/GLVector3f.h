#include <GL/glut.h>

class GLVector3f
{
	public:
		GLfloat x,y,z;
		GLVector3f();
		GLVector3f(GLfloat _x, GLfloat _y, GLfloat _z); 
		~GLVector3f();

		GLvoid add(GLVector3f* _vector);
		GLvoid subtract(GLVector3f* _vector);
		GLfloat dotProduct(GLVector3f* _vector);
		GLvoid crossProduct(GLVector3f* _vector);
		GLvoid set(GLVector3f* _vector);
		GLfloat length();
		GLvoid normalize();
		
};
