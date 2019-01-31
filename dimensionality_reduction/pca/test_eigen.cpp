#include "pca.h"

int main() {
#ifdef TEST_EIGEN
	PCA pca;
	pca.test_eigen();
#else
	cout << "TEST_EIGEN undefined!" << endl;
#endif
	return 0;
}
