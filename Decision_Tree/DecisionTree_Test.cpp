#include "DecisionTree.h"
#include "DecisionTree.cpp"

typedef enum {
	SUNNY,
	OVERCAST,
	RAIN,
} OutLook;

typedef enum {
	HOT,
	MILD,
	COOL,
} Temperature;

typedef enum {
	HIGH,
	NORMAL,
} Humidity;

typedef enum {
	WTRUE,
	WFALSE,
} Windy;

typedef enum {
	YES,
	NO,
} Play;

typedef unsigned char BYTE;

int main() {
	DecisionTree<BYTE> dtree;
	BYTE data[56] = {SUNNY, HOT, HIGH, WFALSE,
			 SUNNY, HOT, HIGH, WTRUE,
			 OVERCAST, HOT, HIGH, WFALSE,
			 RAIN, MILD, HIGH, WFALSE,
			 RAIN, COOL, NORMAL, WFALSE,
			 RAIN, COOL, NORMAL, WTRUE,
			 OVERCAST, COOL, NORMAL, WTRUE,
			 SUNNY, MILD, HIGH, WFALSE,
			 SUNNY, COOL, NORMAL, WFALSE,
			 RAIN, MILD, NORMAL, WFALSE,
			 SUNNY, MILD, NORMAL, WTRUE,
			 OVERCAST, MILD, HIGH, WTRUE,
			 OVERCAST, HOT, NORMAL, WFALSE,
			 RAIN, MILD, HIGH, WTRUE};
	BYTE *dataBuffer = data;
	int label[14] = {NO, NO, YES, YES, YES, NO, YES, NO, YES, YES, YES, YES, YES, NO};
	int nChildren[4] = {3, 3, 2, 2};
	dtree.train(dataBuffer, 14, 4, label, nChildren, 2, C45);
#ifdef PRINT_TREE
	vector<string> attribute_name(4);
	attribute_name[0] = ("OutLook");
	attribute_name[1] = ("Temperature");
	attribute_name[2] = ("Humidity");
	attribute_name[3] = ("Windy");
	vector<vector<string>> attribute_type_name(4);
	attribute_type_name[0].push_back("SUNNY");
	attribute_type_name[0].push_back("OVERCAST");
	attribute_type_name[0].push_back("RAIN");
	attribute_type_name[1].push_back("HOT");
	attribute_type_name[1].push_back("MILD");
	attribute_type_name[1].push_back("COOL");
	attribute_type_name[2].push_back("HIGH");
	attribute_type_name[2].push_back("NORMAL");
	attribute_type_name[3].push_back("WTRUE");
	attribute_type_name[3].push_back("WFALSE");
	vector<string> cluster_name(2);
	cluster_name[0] = "YES";
	cluster_name[1] = "NO";
	dtree.print(attribute_name, attribute_type_name, cluster_name);
#endif

	cout << "predict : SUNNY MILD HIGH WFALSE" << endl;
	BYTE predictData[4] = { SUNNY, MILD, HIGH, WFALSE };
	cout << dtree.predict(predictData);
	return 0;
}
