// BasicOpenCLApplication.cpp : Defines the entry point for the console application.
//CSCM98 Coursework
//Arran Jones - 945187

#include "stdafx.h"
#include "Chrono.h"
#include <math.h>
#include <process.h>
#include <time.h>
#include <immintrin.h>

#include <thread>
#include <mutex>

#define float8 __m256
#define mul8(a,b) _mm256_mul_ps(a,b)
#define div8(a,b) _mm256_div_ps(a,b)
#define add8(a,b) _mm256_add_ps(a,b)
#define sub8(a,b) _mm256_sub_ps(a,b)
#define set8(x) _mm256_set1_ps(x)

#define nbThreads 8

void SaveBMP(char *fname, unsigned char *image, int width, int height, int componentPerPixel=1, int reverseColor=0)
{
	FILE *destination;
    int i,j;
	int *pt;
	char name[512],hdr[0x36];
	unsigned char *imsource=new unsigned char [width*height*3];
	//int al=(ImageSize*3)%4;
	
	if (componentPerPixel==1)
		for (i=0;i<width*height*3;i++)
			imsource[i]=image[i/3];
	else 
		for (i=0;i<width*height*3;i++)
			imsource[i]=image[i];
	if (reverseColor)
		for (j=0;j<height;j++)
			for (i=0;i<width;i++)
			{
				unsigned char aux;
				aux=imsource[3*(i+width*j)];
				imsource[3*(i+width*j)]=imsource[3*(i+width*j)+2];
				imsource[3*(i+width*j)+2]=aux;
			}
	strcpy(name,fname);
	i=(int)strlen(name);
	if (!((i>4)&&(name[i-4]=='.')&&(name[i-3]=='b')&&(name[i-2]=='m')&&(name[i-1]=='p')))
	{
		name[i]='.';
		name[i+1]='b';
		name[i+2]='m';
		name[i+3]='p';
		name[i+4]=0;
	}
	if ((destination=fopen(name, "wb"))==NULL) 
		perror("erreur de creation de fichier\n");
    hdr[0]='B';
    hdr[1]='M';
	pt=(int *)(hdr+2);// file size
	*pt=0x36+width*height*3;
	pt=(int *)(hdr+6);//reserved
	*pt=0x0;
	pt=(int *)(hdr+10);// image address
	*pt=0x36;
	pt=(int *)(hdr+14);// size of [0E-35]
	*pt=0x28;
	pt=(int *)(hdr+0x12);// Image width
	*pt=width;
	pt=(int *)(hdr+0x16);// Image heigth
	*pt=height;
	pt=(int *)(hdr+0x1a);// color planes
	*pt=1;
	pt=(int *)(hdr+0x1c);// bit per pixel
	*pt=24;
	for (i=0x1E;i<0x36;i++) 
		hdr[i]=0;
	fwrite(hdr,0x36,1,destination);
	fwrite (imsource,width*height*3,1,destination);
    fclose(destination);
	delete[] imsource;
}

typedef struct { float real; float im; } complex;

complex add(complex a, complex b)
{
	complex res;
	res.real = a.real + b.real;
	res.im = a.im + b.im;
	return res;
}

complex sub(complex a, complex b)
{
	complex res;
	res.real = a.real - b.real;
	res.im = a.im - b.im;
	return res;
}

complex mul(complex a, complex b)
{
	complex res;
	res.real = a.real*b.real - a.im*b.im;
	res.im = a.real*b.im + a.im*b.real;
	return res;
}

float squaredNorm(complex c)
{
	return c.real*c.real + c.im*c.im;
}

float8 SIMD_calculate(float8 a_real, float8 a_im, float8& z_real, float8& z_im, float8 l_real, float8 l_im) {
	//first sub
	float8 tempSub_real = sub8(a_real, z_real);
	float8 tempSub_im = sub8(a_im, z_im);
	//mul1
	float8 tempMul_real = sub8(mul8(z_real, tempSub_real), mul8(z_im, tempSub_im));
	float8 tempMul_im = add8(mul8(z_real, tempSub_im), mul8(z_im, tempSub_real));
	//mul2
	z_real = sub8(mul8(l_real, tempMul_real), mul8(l_im, tempMul_im));
	z_im = add8(mul8(l_real, tempMul_im), mul8(l_im, tempMul_real));
	float8 squaredNorm = add8(mul8(z_real, z_real), mul8(z_im, z_im));
	return squaredNorm;
}

int Iterate(complex c)
{
	const int max_iterations = 255;
	complex z,a,l;
	a.real = 0.91;
	a.im = 0.;

	z = c;
	l.real = 4;
	l.im = 0;
	int i = 1;
	while (i < max_iterations)
	{
		z = mul(l, mul(z, sub(a, z)));
		if (squaredNorm(z) > 128)
			break;
		i += 2;
	}
	return (min(i, max_iterations));
}

float8 SIMD_Iterate(float8 c_real, float8 c_im) {

	const int max_iterations = 255;
	float8 a_real = set8(0.91);
	float8 a_im = set8(0.);
	boolean test = false;

	float8 z_real = c_real;
	float8 z_im = c_im;
	float8 l_real = set8(4);
	float8 l_im = set8(0);

	int i = 1;
	int breakCount = 0;

	float8 results = set8(1);

	__m256 mk = _mm256_set1_ps(1);

	while (i < max_iterations){
		float8 sqrNorm = SIMD_calculate(a_real, a_im, z_real, z_im, l_real, l_im);

		__m256 mask = _mm256_cmp_ps(sqrNorm, set8(128), _CMP_LT_OS);
		mk = mul8(_mm256_and_ps(mask, _mm256_set1_ps(1)), set8(2));
		results = add8(results, mk);
		
		if (_mm256_movemask_ps(mk)) {
			break;
		}

		//For testing puposes

		/**
		for (int j = 0; j < 8; j++) {
			if (sqrNorm.m256_f32[j] > 128 && results.m256_f32[j] == -1) {
				breakCount++;
				results.m256_f32[j] = i;
				if (breakCount == 8) {
					break;
				}
			}

		}
		if (breakCount == 8) {
			break;
		}
		*/

		i += 2;
	}
		
		results = mul8(set8(2), results);
	return _mm256_min_ps(results, set8(max_iterations));
}

void calculateLines(int j, int dim[2], unsigned char* image, float range[2][2]) {
	for (j; j < dim[1]; j += nbThreads) {
		for (int i = 0;i < dim[0];i += 8) {
			float8 real = set8(0);
			float8 im = set8(0);
			for (int k = 0; k < 8; k++) {
				real.m256_f32[k] = range[0][0] + (i + k + 0.5) * (range[0][1] - range[0][0]) / dim[0]; //Create x coordinates within the range [range[0][0] .. range[0][1]] 
				im.m256_f32[k] = range[1][0] + (j + 0.5) * (range[1][1] - range[1][0]) / dim[1]; //Create x coordinates within the range [range[1][0] .. range[1][1]]
			}
			float8 results = SIMD_Iterate(real, im);
			for (int k = 0; k < 8; k++) {
				image[j * dim[0] + (i + k)] = results.m256_f32[k];
			}
		}
	}
}

void SimpleFractalDrawing(unsigned char *image, int dim[2],float range[2][2])
{
	Chrono c;
	for (int j=0;j<dim[1];j++)
		for (int i=0;i<dim[0];i++)
		{
			complex c;
			c.real=range[0][0]+(i+0.5)*(range[0][1]-range[0][0])/dim[0]; //Create x coordinates within the range [range[0][0] .. range[0][1]] 
			c.im=range[1][0]+(j+0.5)*(range[1][1]-range[1][0])/dim[1]; //Create x coordinates within the range [range[1][0] .. range[1][1]] 
			float f = 2 * Iterate(c);
			if (f > 255.)
				f = 255.;
			image[j*dim[0]+i]=f; 
		}
	c.PrintElapsedTime_ms("time CPU (ms): ");
}

void SimpleFractalDrawingSIMD_MT(unsigned char *image, int dim[2],float range[2][2])
{
	Chrono c;
	std::thread t[nbThreads];
	for (int i = 0;i < nbThreads;i++) {
		t[i] = std::thread(calculateLines, i, dim, image, range);
	}	
	for (int i = 0;i < nbThreads;i++) {
		t[i].join();
	}
	c.PrintElapsedTime_ms("time SIMD_MT (ms): ");
}

int main(int argc, char* argv[])
{
	int dims[2]={1024,1024};
	float range[2][2] = { {-0.003,0.008},{-0.0002,0.0005} };
	float range2[2][2] = { {-1.4,0.6},{-1.1,1.3} };
	unsigned char *image=new unsigned char[dims[0]*dims[1]];
	SimpleFractalDrawing(image,dims,range); //largest 64bit prime
	SaveBMP("fractal.bmp", image, dims[0], dims[1]);
	for (int i = 0; i < dims[0] * dims[1]; i++)
		image[i] = 127; //resetting image to grey
	SimpleFractalDrawingSIMD_MT(image, dims,range);
	SaveBMP("fractalSIMD_MT.bmp",image,dims[0],dims[1]);
	delete[] image;
	return 0; 
}

/*
results:
Computer/processor details: Intel i7 6700k, 4-core/8-thread @ 4.5GHz
CPU 1 thread :		84ms
SIMD_1_Thread:		31ms
SIMD_MT		:		17ms
*/