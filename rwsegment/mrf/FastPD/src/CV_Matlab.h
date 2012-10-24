//#############################################################################
//#
//# CV_Matlab.h:
//#  Contains some useful classes for interfacing with Matlab
//#
//#  @author Nikos Komodakis
//#  
//#############################################################################

#ifndef __CV_MATLAB_H__
#define __CV_MATLAB_H__

#include "mex.h"

//=============================================================================
// CV_Matlab namespace begin
//=============================================================================
//
namespace CV_Matlab 
{

//#############################################################################
//#
//# Classes & types
//#
//#############################################################################

//=============================================================================
// @class   Matrix
//=============================================================================
//
template <class Type>
class Matrix
{
	public:

		Matrix( mxArray *matrix = NULL )
		{
			reset(matrix);
		}

		void reset( mxArray *matrix )
		{
			if (matrix)
			{
				m_array = matrix;
				m_data  = (Type *)mxGetData( matrix );
				m_sizeX = mxGetN( matrix );
				m_sizeY = mxGetM( matrix );
			}
			else
			{
				m_array = NULL;
				m_data  = NULL;
				m_sizeX = 0;
				m_sizeY = 0;
			}
		}

		const Type operator()( int x, int y ) const
		{
			return m_data[ x * m_sizeY + y ];
		}

		Type &operator()( int x, int y )
		{
			return m_data[ x * m_sizeY + y ];
		}

		int width( void )
		{
			return m_sizeX;
		}

		int height( void )
		{
			return m_sizeY;
		}

		Type *data( void )
		{
			return m_data;
		}

		mxArray *nativeArray( void )
		{
			return m_array;
		}

	private:

		mxArray *m_array;
		Type    *m_data;
		int      m_sizeX;
		int      m_sizeY;
};

//=============================================================================
// @class   String
//=============================================================================
//
class String
{
	public:

		String( mxArray *str )
		{
			m_mxstr = str;

			int buflen = (mxGetM(m_mxstr) * mxGetN(m_mxstr)) + 1;
			m_str = new char[buflen];
			mxGetString( m_mxstr, m_str, buflen);
		}

		~String( void )
		{
			delete [] m_str;
		}

		char *c_str( void )
		{
			return m_str;
		};

	private:

		// Assignment or copying not allowed
		//
		String( const String & );
		String &operator=( const String & );

		char *m_str;
		mxArray *m_mxstr;
};

//=============================================================================
// CV_Matlab namespace end
//=============================================================================
//
} 

#endif /* __CV_MATLAB_H__ */

//#############################################################################
//#
//# EOF
//#
//#############################################################################

