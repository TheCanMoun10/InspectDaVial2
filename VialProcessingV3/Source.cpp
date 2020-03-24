#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

using namespace cv;
using namespace std;

struct Vialfeatures
{
	int contourIndex;
	int area;
	int perimeter;
	float circularity;
	bool hasCrack;
	float elongation;
};

void Good()
{
	//Load the Image needed:
	Mat vial = imread("C:\\Users\\...\\VialsForProject\\DarkRoomVial.png"); //Location of the files (C://Users//...//VialsForProject//xx.png

	if (vial.empty()) cout << "Failed loading image" << endl;
	else cout << "Image Loaded succesfully" << endl;

	//Defining Region Of Interest (ROI):
	Rect ROI = Rect(115, 43, 106, 312); // ROI of interest for DarkRoomVials (115,43,106,312) , Found using ImageJ (For other vials: 198, 108, 139, 372)
	Mat vial_ROI = vial(ROI);

	//For the Good Vial:
	Mat Grayscale = Mat(vial_ROI.size(), CV_8U);
	Mat vial_Blur = Mat(vial_ROI.size(), CV_8U);
	Mat vial_Subt = Mat(vial_ROI.size(), CV_8U);
	Mat vial_Thresh = Mat(vial_ROI.size(), CV_8U);
	Mat Contour = Mat(vial_ROI.size(), CV_8U, Scalar(255,255,255));
	Mat ContourIMG = Mat(vial_ROI.size(), CV_8U, Scalar(255, 255, 255));


	

	//Thresholding:
	medianBlur(vial_ROI, vial_Blur, 41); //Blurring image to get rid of uneven lighting.
	subtract(vial_Blur, vial_ROI, vial_Subt); //Subtracting background to only get circumference of vial.

	//Before threshold can be applied a color-to-grayscale is needed using cvtColor:
	cvtColor(vial_Subt, Grayscale, COLOR_BGR2GRAY);

	//Now the most optimal threshold is found, using threshold command:
	threshold(Grayscale, vial_Thresh, 0, 255, THRESH_BINARY | THRESH_OTSU);

	//Morpholoy application Good Vial:
	//Morphology matrices:
	Mat vial_morph1 = Mat(vial_ROI.size(), CV_8U);
	Mat vial_morph2 = Mat(vial_ROI.size(), CV_8U);
	Mat vial_morph3 = Mat(vial_ROI.size(), CV_8U);
	Mat vial_morph4 = Mat(vial_ROI.size(), CV_8U);
	Mat ElemOpen = getStructuringElement(MORPH_RECT, Size(1, 4));
	Mat ElemClose = getStructuringElement(MORPH_ELLIPSE, Size(3, 6));
	Mat ElemDilate = getStructuringElement(MORPH_RECT, Size(7, 7));
	Mat ElemErode = getStructuringElement(MORPH_RECT, Size(6, 6));
	morphologyEx(vial_Thresh, vial_morph1, MORPH_DILATE, ElemDilate);
	morphologyEx(vial_morph1, vial_morph2, MORPH_CLOSE, ElemClose);
	morphologyEx(vial_morph2, vial_morph3, MORPH_OPEN, ElemOpen);
	morphologyEx(vial_morph3, vial_morph4, MORPH_ERODE, ElemErode);


	//Contouring:
	vector<vector<Point>> Gcontours;
	vector<Vec4i> hierarchy;

	findContours(vial_morph4, Gcontours, hierarchy, RETR_TREE, CHAIN_APPROX_NONE);
	drawContours(Contour, Gcontours, -1, Scalar(0, 0, 0), 1);

	vector<Vialfeatures> featVec;

	//Looping through all contours, for now only external contours (hierarchy-parent = -1):
	//Save contour index and features in a vector:

	for (int i = 0; i < Gcontours.size(); i++)
	{
		if (hierarchy[i][3] == -1)
		{
			Vialfeatures G;
			G.contourIndex = i;
			G.area = contourArea(Gcontours[i]);
			G.perimeter = arcLength(Gcontours[i], true);
			G.circularity = (4 * 3.14 * G.area) / pow(G.perimeter, 2);
			G.hasCrack = (hierarchy[i][2] == -1) ? false : true;

			RotatedRect box = minAreaRect(Gcontours[i]);
			G.elongation = max(box.size.width / box.size.height, box.size.height / box.size.width);

			cout << "G.Area:" << G.area << "; G.Perimeter:" << G.perimeter << "; G.Circularity:" << G.circularity << "; G.Cracks:" << G.hasCrack << "; G.Elongation:" << G.elongation << endl;

			featVec.push_back(G);
		}
	}

	for (int i = 0; i < featVec.size(); i++)
	{
		if (featVec[i].area < 5000)
		{
			drawContours(ContourIMG, Gcontours, featVec[i].contourIndex, Scalar(0, 150, 0), 1);
		}

		if (featVec[i].elongation > 1000)
		{
			drawContours(ContourIMG, Gcontours, featVec[i].contourIndex, Scalar(255, 0, 0), 1);
		}

		if (featVec[i].area < 10000 && featVec[i].hasCrack == true)
		{
			drawContours(ContourIMG, Gcontours, featVec[i].contourIndex, Scalar(255, 255, 0), 1);
		}

	}


	//Results with Good vial:
	//imshow("Original Image", vial);
	//imshow("Background", vial_Subt);
	imshow("Region of Interest", vial_ROI);
	//imshow("Threshold", vial_Thresh);
	//imshow("Morphology", vial_morph4);
	imshow("Contours", ContourIMG);
}

void Bad()
{
	Mat BadVial = imread("C:\\Users\\...\\VialsForProject\\DarkRoomCrack.png");



	if (BadVial.empty()) cout << "Failed loading Image" << endl;
	else cout << "Image Loaded succesfully" << endl;

	//Defining Region Of Interest (ROI):
	Rect ROI = Rect(115, 43, 106, 312); // ROI of interest for DarkRoomVials, Found using ImageJ (For other vials: 198, 108, 139, 372)
	Mat BadVial_ROI = BadVial(ROI);

	//Bad Vial:
	Mat GrayscaleBad = Mat(BadVial_ROI.size(), CV_8U);
	Mat vial_BlurBad = Mat(BadVial_ROI.size(), CV_8U);
	Mat vial_SubtBad = Mat(BadVial_ROI.size(), CV_8U);
	Mat vial_ThreshBad = Mat(BadVial_ROI.size(), CV_8U);
	Mat ContourBad = Mat(BadVial_ROI.size(), CV_8U, Scalar(255,255,255));
	Mat ContourIMGBad = Mat(BadVial_ROI.size(), CV_8UC3, Scalar(255, 255, 255));


	//Thresholding:
	medianBlur(BadVial_ROI, vial_BlurBad, 31); //Blurring image to get rid of uneven lighting. (Number needs to be uneven)
	subtract(vial_BlurBad, BadVial_ROI, vial_SubtBad); //Subtracting background to only get circumference of vial.

	//Before threshold can be applied a color-to-grayscale is needed using cvtColor:
	cvtColor(vial_SubtBad, GrayscaleBad, COLOR_BGR2GRAY);

	//Now the most optimal threshold is found, using threshold command:
	threshold(GrayscaleBad, vial_ThreshBad, 0, 255, THRESH_BINARY | THRESH_OTSU);

	//Morpholoy application Good Vial:
	//Morphology matrices:
	Mat Badvial_morph1 = Mat(BadVial_ROI.size(), CV_8U);
	Mat Badvial_morph2 = Mat(BadVial_ROI.size(), CV_8U);
	Mat Badvial_morph3 = Mat(BadVial_ROI.size(), CV_8U);
	Mat BadElem1 = getStructuringElement(MORPH_RECT, Size(2, 5));
	Mat BadElem2 = getStructuringElement(MORPH_ELLIPSE, Size(4, 6));
	Mat BadElem3 = getStructuringElement(MORPH_RECT, Size(1, 1));
	morphologyEx(vial_ThreshBad, Badvial_morph1, MORPH_DILATE, BadElem1);
	morphologyEx(Badvial_morph1, Badvial_morph2, MORPH_CLOSE, BadElem2);
	morphologyEx(Badvial_morph2, Badvial_morph3, MORPH_ERODE, BadElem3);


	//Contouring:
	vector<vector<Point>> Bcontours;
	vector<Vec4i> Bhierarchy;

	findContours(Badvial_morph3, Bcontours, Bhierarchy, RETR_TREE, CHAIN_APPROX_NONE);
	drawContours(ContourBad, Bcontours, -1, Scalar(0, 0, 0), 1);

	vector<Vialfeatures> BfeatVec;

	//Looping through all contours, for now only external contours (hierarchy-parent = -1):
	//Save contour index and features in a vector:

	for (int i = 0; i < Bcontours.size(); i++)
	{
		if (Bhierarchy[i][3] == -1)
		{
			Vialfeatures B;
			B.contourIndex = i;
			B.area = contourArea(Bcontours[i]);
			B.perimeter = arcLength(Bcontours[i], true);
			B.circularity = (4 * 3.14 * B.area) / pow(B.perimeter, 2);
			B.hasCrack = (Bhierarchy[i][2] == -1) ? false : true;

			RotatedRect box = minAreaRect(Bcontours[i]);
			B.elongation = max(box.size.width / box.size.height, box.size.height / box.size.width);

			cout << "Area:" << B.area << "; Perimeter:" << B.perimeter << "; Circularity:" << B.circularity << "; Cracks:" << B.hasCrack << "; Elongation:" << B.elongation << endl;

			BfeatVec.push_back(B);
		}
	}

	for (int i = 0; i < BfeatVec.size(); i++)
	{
		if (BfeatVec[i].area > 2500)
		{
			drawContours(ContourIMGBad, Bcontours, BfeatVec[i].contourIndex, Scalar(0, 0, 0), 1);
			cout << "Detected Vial";
		}

		if (BfeatVec[i].perimeter < 30 || BfeatVec[i].perimeter < 400)
		{
			drawContours(ContourIMGBad, Bcontours, BfeatVec[i].contourIndex, Scalar(0, 0, 255), 1);
			cout << "Defects highlighted red \n";
		}

		/*
		if (BfeatVec[i].circularity > 0.4 && BfeatVec[i].perimeter < 30)
		{
			drawContours(ContourIMGBad, Bcontours, BfeatVec[i].contourIndex, Scalar(0, 0, 255), 1);
		}
		*/
	}

	//Results Bad Vial:
	//imshow("Bad Vial -Original", BadVial);
	imshow("Bad Vial - Region of Interest", BadVial_ROI);
	//imshow("Bad Vial - Threshold", vial_ThreshBad);
	//imshow("Bad Vial - Morphology", Badvial_morph3);
	//imshow(Bad  )
	imshow("Bad Vial - Contours", ContourIMGBad);
	//imshow("Background - Bad", vial_SubtBad);


}

void main() 
{
	Good();
	Bad();
	

	waitKey(0);
}
