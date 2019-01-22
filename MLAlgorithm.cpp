#include <Python.h>
#include "Kmeans.h"
#include "dbscan.h"
#include "knn.h"

//Constructor of algorithms
Kmeans km;
Dbscan dbscan;
Knn knn;

//Kmeans
//
static PyObject *py_kmeans(PyObject *self, PyObject *args) {
	int nClusters;
	PyObject *dataBuffer;
	PyObject *labelBuffer;

	//Get the passed Python object
	if (!PyArg_ParseTuple(args, "i|O|O", &nClusters, &dataBuffer, &labelBuffer)) {
		return NULL;
	}

	//Extract the buffer information
	Py_buffer dataView, labelView;

	//Extract data buffer
	if (PyObject_GetBuffer(dataBuffer, &dataView, PyBUF_ANY_CONTIGUOUS | PyBUF_FORMAT) == -1) {
		return NULL;
	}

	if (dataView.ndim != 2) {
		PyErr_SetString(PyExc_TypeError, "Arg 2 expected a 2-dimensional array");
		PyBuffer_Release(&dataView);
		return NULL;
	}
	if (strcmp(dataView.format, "d") != 0 && strcmp(dataView.format, "i") != 0 && strcmp(dataView.format, "f") != 0 && strcmp(dataView.format, "l") != 0) {
		PyErr_SetString(PyExc_TypeError, "Arg 2 expected an array of numbers");
		PyBuffer_Release(&dataView);
		return NULL;
	}

	//Extract label buffer
	if (PyObject_GetBuffer(labelBuffer, &labelView, PyBUF_ANY_CONTIGUOUS | PyBUF_FORMAT) == -1) {
		return NULL;
	}

	if (labelView.ndim != 1) {
		PyErr_SetString(PyExc_TypeError, "Arg 3 expected a 1-dimensional array");
		PyBuffer_Release(&dataView);
		PyBuffer_Release(&labelView);
		return NULL;
	}

	if (strcmp(labelView.format, "d") != 0 && strcmp(labelView.format, "i") != 0 && strcmp(labelView.format, "f") != 0 && strcmp(labelView.format, "l") != 0) {
		PyErr_SetString(PyExc_TypeError, "Arg 3 expected an array of numbers");
		PyBuffer_Release(&dataView);
		PyBuffer_Release(&labelView);
		return NULL;
	}

	if (dataView.shape[0] != labelView.shape[0]) {
		PyBuffer_Release(&dataView);
		PyBuffer_Release(&labelView);
		return NULL;
	}

	//Call function in Kmeans
	km.clustering(nClusters, (double *)dataView.buf, dataView.shape[0], dataView.shape[1], (int *)labelView.buf);

	PyBuffer_Release(&dataView);
	PyBuffer_Release(&labelView);
	return Py_BuildValue("i", 0);
}

//DBSCAN
//
static PyObject *py_dbscan(PyObject *self, PyObject *args) {
	int dist, minPts;
	PyObject *dataBuffer;
	PyObject *labelBuffer;

	//Get the Python object passed in
	if (!PyArg_ParseTuple(args, "i|i|O|O", &dist, &minPts, &dataBuffer, &labelBuffer)) {
		return NULL;
	}

	//Extract the buffer information
	Py_buffer dataView, labelView;

	//Extract data buffer
	if (PyObject_GetBuffer(dataBuffer, &dataView, PyBUF_ANY_CONTIGUOUS | PyBUF_FORMAT) == -1) {
		return NULL;
	}

	if (dataView.ndim != 2) {
		PyErr_SetString(PyExc_TypeError, "Arg 3 expected a 2-dimentional array");
		PyBuffer_Release(&dataView);
		return NULL;
	}	
	if (strcmp(dataView.format, "d") != 0 && strcmp(dataView.format, "i") != 0 && strcmp(dataView.format, "f") != 0 && strcmp(dataView.format, "l") != 0) {
		PyErr_SetString(PyExc_TypeError, "Arg 3 expected an array of numbers");
		PyBuffer_Release(&dataView);
		return NULL;
	}

	//Extract label buffer
	if (PyObject_GetBuffer(labelBuffer, &labelView, PyBUF_ANY_CONTIGUOUS | PyBUF_FORMAT) == -1) {
		return NULL;
	}

	if (labelView.ndim != 1) {
		PyErr_SetString(PyExc_TypeError, "Arg 4 expected a 1-dimensional array");
		PyBuffer_Release(&dataView);
		PyBuffer_Release(&labelView);
		return NULL;
	}
	
	if (strcmp(labelView.format, "d") != 0 && strcmp(labelView.format, "i") != 0 && strcmp(labelView.format, "f") != 0 && strcmp(labelView.format, "l") != 0) {
		PyErr_SetString(PyExc_TypeError, "Arg 4 expected an array of integers");
		PyBuffer_Release(&dataView);
		PyBuffer_Release(&labelView);
		return NULL;
	}

	if (dataView.shape[0] != labelView.shape[0]) {
		PyBuffer_Release(&dataView);
		PyBuffer_Release(&labelView);
		return NULL;
	}

	//Call function in dbscan
	int nClusters = dbscan.DBSCAN(dist, minPts, (double *)dataView.buf, dataView.shape[0], dataView.shape[1], (int *)labelView.buf);

	PyBuffer_Release(&dataView);
	PyBuffer_Release(&labelView);

	return Py_BuildValue("i", nClusters);
}

static PyObject *py_dbscan_isNoiseLabel(PyObject *self, PyObject *args) {
	int label;
	if (!PyArg_ParseTuple(args, "i", &label)) {
		return NULL;
	}
	return Py_BuildValue("i", dbscan.isNoiseLabel(label));
}


//KNN
//
static PyObject *py_knn_train(PyObject *self, PyObject *args) {
	PyObject *dataBuffer;
	PyObject *labelBuffer;

	//Get the Python object passed in
	if (!PyArg_ParseTuple(args, "O|O", &dataBuffer, &labelBuffer)) {
		return NULL;
	}

	//Extract the buffer information
	Py_buffer dataView, labelView;

	//Extract data buffer
	if (PyObject_GetBuffer(dataBuffer, &dataView, PyBUF_ANY_CONTIGUOUS | PyBUF_FORMAT) == -1) {
		return NULL;
	}

	if (dataView.ndim != 2) {
		PyErr_SetString(PyExc_TypeError, "Arg 1 expected a 2-dimentional array");
		PyBuffer_Release(&dataView);
		return NULL;
	}	
	if (strcmp(dataView.format, "d") != 0 && strcmp(dataView.format, "i") != 0 && strcmp(dataView.format, "f") != 0 && strcmp(dataView.format, "l") != 0) {
		PyErr_SetString(PyExc_TypeError, "Arg 1 expected an array of numbers");
		PyBuffer_Release(&dataView);
		return NULL;
	}

	//Extract label buffer
	if (PyObject_GetBuffer(labelBuffer, &labelView, PyBUF_ANY_CONTIGUOUS | PyBUF_FORMAT) == -1) {
		return NULL;
	}

	if (labelView.ndim != 1) {
		PyErr_SetString(PyExc_TypeError, "Arg 2 expected a 1-dimensional array");
		PyBuffer_Release(&dataView);
		PyBuffer_Release(&labelView);
		return NULL;
	}
	
	if (strcmp(labelView.format, "d") != 0 && strcmp(labelView.format, "i") != 0 && strcmp(labelView.format, "f") != 0 && strcmp(labelView.format, "l") != 0) {
		PyErr_SetString(PyExc_TypeError, "Arg 2 expected an array of integers");
		PyBuffer_Release(&dataView);
		PyBuffer_Release(&labelView);
		return NULL;
	}

	if (dataView.shape[0] != labelView.shape[0]) {
		PyBuffer_Release(&dataView);
		PyBuffer_Release(&labelView);
		return NULL;
	}

	//Call function in dbscan
	knn.train((double *)dataView.buf, dataView.shape[0], dataView.shape[1], (int *)labelView.buf);
		
	PyBuffer_Release(&dataView);
	PyBuffer_Release(&labelView);

	return Py_BuildValue("i", 0);
}

static PyObject *py_knn_predict(PyObject *self, PyObject *args) {
	PyObject *pointBuffer;
	int k;
	//Get the Python object passed in
	if (!PyArg_ParseTuple(args, "i|O", &k, &pointBuffer)) {
		return NULL;
	}

	//Extract the buffer information
	Py_buffer pointView;

	//Extract data buffer
	if (PyObject_GetBuffer(pointBuffer, &pointView, PyBUF_ANY_CONTIGUOUS | PyBUF_FORMAT) == -1) {
		return NULL;
	}

	if (pointView.ndim != 1) {
		PyErr_SetString(PyExc_TypeError, "Arg 1 expected a 1-dimentional array");
		PyBuffer_Release(&pointView);
		return NULL;
	}	
	if (strcmp(pointView.format, "d") != 0 && strcmp(pointView.format, "i") != 0 && strcmp(pointView.format, "f") != 0 && strcmp(pointView.format, "l") != 0) {
		PyErr_SetString(PyExc_TypeError, "Arg 1 expected an array of numbers");
		PyBuffer_Release(&pointView);
		return NULL;
	}

	//Call function in dbscan
	int predict_label = knn.predict(k, (double *)pointView.buf, pointView.shape[0]);
	if (predict_label == KNN_NODATA_ERROR) {
		PyErr_SetString(PyExc_TypeError, "Call knn_train first!");
		PyBuffer_Release(&pointView);
		return NULL;
	}
	if (predict_label == KNN_DIMENSION_ERROR) {
		PyErr_SetString(PyExc_TypeError, "Expected a point with the same dimension as the trained data!");
		PyBuffer_Release(&pointView);
		return NULL;
	}

	PyBuffer_Release(&pointView);

	return Py_BuildValue("i", predict_label);
}

static PyMethodDef MLAlgorithmFunc[] = {
	{"kmeans", py_kmeans, METH_VARARGS, "clustering by kmeans"},
	{"dbscan", py_dbscan, METH_VARARGS, "clustering by dbscan"},
	{"dbscan_isNoiseLabel", py_dbscan_isNoiseLabel, METH_VARARGS, "Returns true if it's a noise label in dbscan"},
	{"knn_train", py_knn_train, METH_VARARGS, "KNN train data"},
	{"knn_predict", py_knn_predict, METH_VARARGS, "KNN predict data"},
	{NULL, NULL, 0, NULL},
};

static struct PyModuleDef MLAlgorithm = {
	PyModuleDef_HEAD_INIT,
	"MLAlgorithm",
	"",
	-1,
	MLAlgorithmFunc
};

PyMODINIT_FUNC PyInit_MLAlgorithm(void) {
	return PyModule_Create(&MLAlgorithm);
}
