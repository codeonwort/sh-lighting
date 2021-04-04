#include <math.h>
#include <stdio.h>
#include <time.h>
#include <functional>
#include <vector>

// Advanced Lighting and Materials with Shaders (Kelly Dempski, Emmanuel Viale)
// Chapter 8. Spherical Harmonic Lighting

// f(s) : 2D function defined on unit sphere
// reconstruction of f(s)
// {
//     x = 0
//     for (l = 0 to n)
//         for(m = -l to l)
//             x += coeff(L, m) * SHbasis(L, m, s)
//     return x
// }
// coeff(L, m) = { integrate (f(s) * SHbasis(L, m)) over S }
//            ~= MC integration of the integrand

struct SHVec3
{
	SHVec3(double x0, double y0, double z0) : x(x0), y(y0), z(z0) {}
	SHVec3() : x(0.0), y(0.0), z(0.0) {}
	double x, y, z;
};
struct SHSample
{
	SHVec3 sph;
	SHVec3 vec;
	std::vector<double> coeff;
};

double rnd()
{
	return (double)rand() / RAND_MAX;
}

#define max(a, b) ((a > b) ? (a) : (b))

int doubleFactorial(int x)
{
	if (x == 0 || x == -1) return 1;
	int result = x;
	while ((x -= 2) > 0) result *= x;
	return result;
}

// #todo: cache it
// Associated Legendre polynomial (recursive ver.)
double ALPStd(float x, int L, int m)
{
	if (L == m)
	{
		int sgn = (m & 1) ? (-1) : 1;
		return sgn * doubleFactorial(2 * m - 1) * pow(sqrt(1 - x * x), m);
	}
	if (L == m + 1)
	{
		return x * (2 * m + 1) * ALPStd(x, m, m);
	}
	double num = (x * (2 * L - 1) * ALPStd(x, L - 1, m)) - ((L + m - 1) * ALPStd(x, L - 2, m));
	double denom = L - m;
	return num / denom;
}

int factorial(int x)
{
	if (x == 0 || x == 1) return 1;
	int result = x;
	while (x --> 1) result *= x;
	return result;
}

double evaluateK(int L, int m)
{
	double num = (2.0 * L + 1.0) * factorial(L - m);
	double denom = 4 * M_PI * factorial(L + m);
	return num / denom;
}

// SH coeff. is multiplication of K-factor, ALPStd, and some cos/sin
double evaluateSH(int L, int m, double theta, double phi)
{
	double SH = 0.0;
	if (m == 0)    SH = evaluateK(L, 0) * ALPStd(cos(theta), L, 0);
	else if(m > 0) SH = M_SQRT2 * evaluateK(L, m) * cos(m * phi) * ALPStd(cos(theta), L, m);
	else           SH = M_SQRT2 * evaluateK(L, -m) * sin(-m * phi) * ALPStd(cos(theta), L, -m);
	return SH;
}

// Initialize SH samples
void sphericalStratifiedSampling(std::vector<SHSample>& outSamples, int sqrtNumSamples, int nBands)
{
	outSamples.resize(sqrtNumSamples * sqrtNumSamples);

	const double invNumSamples = 1.0 / sqrtNumSamples;
	int i = 0;

	for (int a = 0; a < sqrtNumSamples; ++a)
	{
		for (int b = 0; b < sqrtNumSamples; ++b)
		{
			double x = (a + rnd()) * invNumSamples;
			double y = (b + rnd()) * invNumSamples;

			double theta = 2.0 * acos(sqrt(1.0 - x));
			double phi = 2.0 * M_PI * y;

			outSamples[i].sph = { theta, phi, 1.0 };
			outSamples[i].vec = { sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta) };

			//outSamples[i].coeff.clear();
			outSamples[i].coeff.resize(nBands * nBands);
			for (int L = 0; L < nBands; ++L)
			{
				for(int m = -L; m <= L; ++m)
				{
					int index = L * (L + 1) + m;
					outSamples[i].coeff[index] = evaluateSH(L, m, theta, phi);
					//outSamples[i].coeff.push_back(evaluateSH(L, m, theta, phi));
				}
			}

			++i;
		}
	}
}

using SphericalFunction = std::function<double(double theta, double phi)>;

// Projects a function to SH basis and returns its SH coefficients.
void SHProjectSphericalFunction(SphericalFunction myFn, const std::vector<SHSample>& samples, std::vector<double>& outResult)
{
	const int numSamples = samples.size();
	const int numCoeffs = samples[0].coeff.size();

	double dWeight = 4.0 * M_PI;
	double factor = dWeight / numSamples;

	outResult.resize(numCoeffs, 0.0);

	for (int i = 0; i < numSamples; ++i)
	{
		double theta = samples[i].sph.x;
		double phi = samples[i].sph.y;

		for (int n = 0; n < numCoeffs; ++n)
		{
			outResult[n] += myFn(theta, phi) * samples[i].coeff[n];
		}
	}

	for (int i = 0; i < numCoeffs; ++i)
	{
		outResult[i] = outResult[i] * factor;
	}
}

////////////////////////////////////////////////////////////////////////////////////////////////////

double testLight(double theta, double phi)
{
	return (max(0, 5 * cos(theta) - 4) + (max(0, -4 * sin(theta - M_PI) * cos(phi - 2.5) - 3)));
}

// Evaluate testLight with SH basis to test accuracy
double testLightSH(double theta, double phi, const std::vector<double>& myCoeff, int nBands)
{
	double result = 0.0;
	int i = 0;

	for (int L = 0; L < nBands; ++L)
	{
		for(int m = -L; m <= L; ++m)
		{
			result += myCoeff[i] * evaluateSH(L, m, theta, phi);
			++i;
		}
	}

	return result;
}


#define VERBOSE 0

int main()
{
	// 0. Initialization
	srand(time(NULL));

	const int nSamples = 100 * 100;
	const int nBands = 5;
	std::vector<SHSample> shSamples;
	std::vector<double> myCoeff;

	// 1. Get SH Samples
	{
		sphericalStratifiedSampling(shSamples, sqrt(nSamples), nBands);

		printf("# of sh samples: %d\n", nSamples);
		printf("# of sh bands: %d\n", nBands);
#if VERBOSE
		puts("=== DEBUG: SH Samples ===");
		for (int i = 0; i < nSamples; ++i)
		{
			SHSample& sam = shSamples[i];
			printf("sph = (%lf, %lf, %lf)\n", sam.sph.x, sam.sph.y, sam.sph.z);
			printf("vec = (%lf, %lf, %lf)\n", sam.vec.x, sam.vec.y, sam.vec.z);
			printf("coeff = (");
			for (int j = 0; j < sam.coeff.size(); ++j)
			{
				printf("%lf ", sam.coeff[j]);
			}
			printf(")\n");
		}
#endif
	}

	// 2. Project my light into SH basis
	{
		SHProjectSphericalFunction(testLight, shSamples, myCoeff);

		printf("# of coeff = %d\n", (int)myCoeff.size());
#if VERBOSE
		puts("=== DEBUG: Projected function ===");
		printf("myCoeff = (");
		for (int i = 0; i < myCoeff.size(); ++i)
		{
			printf("%lf ", myCoeff[i]);
		}
		puts(")");
#endif
	}

	// 3. Evaluate my light with SH basis to check the discrepancy
	// #todo: Too large errors than expected? :/
	{
		puts("=== TEST: Evaluate light - Original vs SH basis ===");
		const int nTest = 10;
		double avgErr = 0.0;
		for (int i = 0; i < nTest; ++i)
		{
			double theta = 2.0 * M_PI * rnd();
			double phi = 2.0 * M_PI * rnd();
			double x1 = testLight(theta, phi);
			double x2 = testLightSH(theta, phi, myCoeff, nBands);

			printf("diff of f(%.3lf, %.3lf) -> %lf (%lf - (%lf))\n", theta, phi, x1 - x2, x1, x2);

			avgErr += abs(x1 - x2);
		}
		printf("Average error: %lf\n", avgErr / nTest);
	}

	return 0;
}

