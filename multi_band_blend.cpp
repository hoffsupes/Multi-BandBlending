
#include <opencv2/highgui/highgui.hpp>
#include <algorithm>
#include <opencv2/core/core.hpp>
#include <iostream>
#include <vector>
#include <cstdarg>
#include "opencv2/opencv.hpp"
#include "fstream"
#include "Ctracker.h"
#include <dirent.h>
#include <math.h>
#include <time.h>
#include	<opencv2/videostab.hpp>
#include 	<opencv2/features2d.hpp>
#include <opencv2/videostab/global_motion.hpp>
#include <opencv2/videostab/outlier_rejection.hpp>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <opencv2/stitching.hpp>
#include <opencv2/stitching/detail/blenders.hpp>

using namespace std;
using namespace cv;
using	namespace	cv::videostab;
using namespace cv::detail;

void regstECC(Mat src,Mat dst,Mat &U,const int warp_mode) // src:: reference/template frame dst::inst frame
{
// Convert images to gray scale;
Mat im1_gray, im2_gray;
	
cvtColor(src, im1_gray, CV_BGR2GRAY);
cvtColor(dst, im2_gray, CV_BGR2GRAY);
	
 
// Define the motion model
// const int warp_mode = MOTION_AFFINE;
 
// Set a 2x3 or 3x3 warp matrix depending on the motion model.
Mat warp_matrix;
 
// Initialize the matrix to identity
if ( warp_mode == MOTION_HOMOGRAPHY )
    warp_matrix = Mat::eye(3, 3, CV_32F);
else
    warp_matrix = Mat::eye(2, 3, CV_32F);
 
// Specify the number of iterations.
int number_of_iterations = 100;
 
// Specify the threshold of the increment
// in the correlation coefficient between two iterations
double termination_eps = 1e-10;
 
// Define termination criteria
TermCriteria criteria (TermCriteria::COUNT+TermCriteria::EPS, number_of_iterations, termination_eps);
 
// Run the ECC algorithm. The results are stored in warp_matrix.
findTransformECC(
                 im1_gray,
                 im2_gray,
                 warp_matrix,
                 warp_mode,
                 criteria
             );
 
// Storage for warped image.
 
if (warp_mode != MOTION_HOMOGRAPHY)
    // Use warpAffine for Translation, Euclidean and Affine
    warpAffine(dst,U, warp_matrix, src.size(), INTER_LINEAR);
else
    // Use warpPerspective for Homography
    warpPerspective (dst, U, warp_matrix, src.size(),INTER_LINEAR);
}

Mat getreducemat(Mat mgpts,Mat scpts,Mat a13,Mat a46)
{

	Mat red,A13,A46,newmg,newsc,XX,YY;
	
	Mat SCP[] = {scpts,Mat::ones(scpts.rows,1,scpts.type()) };
	hconcat(SCP,2,newsc);
	
	repeat(a13, 1,scpts.rows, A13);	A13 = A13.t();
	repeat(a46,1,scpts.rows, A46);	A46 = A46.t();
	
	multiply(A13,newsc,A13);
	multiply(A46,newsc,A46);
	
	reduce(A13,XX,1,CV_REDUCE_SUM);
	reduce(A46,YY,1,CV_REDUCE_SUM);

	Mat rree[] = {XX,YY};
	hconcat(rree,2,red);
	
	red = mgpts - red;
	
	return red;
	
}

Mat get_affine_matrix(Mat movpoints,Mat srcpoints)			// srcpoints = moving frame ||| movpoints = static frame
// Mat get_affine_matrix(Mat srcpoints,Mat movpoints)			// srcpoints = moving frame ||| movpoints = static frame
{
float epsilon = 0.01;
Mat ontm = Mat::ones(srcpoints.rows, 1, srcpoints.type());
Mat A,X,Y,a13, a46;
	
Mat mtarr[] = {srcpoints, ontm};
hconcat(mtarr, 2 , A);

movpoints.col(0).copyTo(X);
movpoints.col(1).copyTo(Y);

/// FOR TESTING PURPOSES ONLY REMOVE LATER	
ofstream ffcout;
ffcout.open("ress.txt",ios::out);
	
 a13 = (A.t() * A).inv(DECOMP_CHOLESKY) * (A.t() * X);		// Check for SVD parameters for accurate warp
 a46 = (A.t() * A).inv(DECOMP_CHOLESKY) * (A.t() * Y);

// a13 = (A.inv(DECOMP_SVD)) * X;
// a46 = (A.inv(DECOMP_SVD)) * Y;
float J = 0,Jold = 0,delJ = 0,olddelJ = 0;
float thep = 100;
int cco = 0;	
	Mat Xresidue, Yresidue,XW,YW;

	// loop condition to be put here
	do{

	Mat redu =  getreducemat(movpoints,srcpoints,a13,a46);
	Mat W;	
	divide(1,abs(redu) + epsilon,W);
	
	multiply(A.col(0),W.col(0),A.col(0));
	multiply(A.col(1),W.col(1),A.col(1));
	
	multiply(X,W.col(0),X);
	multiply(Y,W.col(1),Y);
	
 	a13 = (A.t() * A).inv(DECOMP_CHOLESKY) * (A.t() * X);		// Check for SVD parameters for accurate warp
 	a46 = (A.t() * A).inv(DECOMP_CHOLESKY) * (A.t() * Y);	
	
	pow(redu,2,redu);
		
	Jold = J;		
	J = sum(redu.col(0))[0] + sum(redu.col(1))[0];
	olddelJ = delJ;
	delJ = abs(J - Jold);
		
	cco++;	
	}while(	(delJ > thep) && (cco > 0) & (( (abs(delJ - olddelJ) == 0)) || (abs(delJ - olddelJ) <= 1.5)) ) ;

Mat affine_matrix; Mat tmpone  = Mat::zeros(1,3,CV_32F); tmpone.at<float>(2) = 1;
affine_matrix.push_back(a13.t());
affine_matrix.push_back(a46.t());
//affine_matrix.push_back(tmpone);

	ffcout.close();
	
return affine_matrix;
}

Mat warp_frame(Mat mov, Mat W)
{
	mov.convertTo(mov, CV_32F);
	Mat warped_fram = Mat::zeros(mov.size(), mov.type());
	
	for(int c = 0; c< warped_fram.channels(); c++)
	{
	
	for(int y = 0;y< warped_fram.rows; y++)				// Y value
	{
		for(int x = 0;x < warped_fram.cols;x++)
		{
		Mat tmpcor = Mat::ones(3,1,CV_32F); tmpcor.at<float>(0) = x; tmpcor.at<float>(1) = y;
		Mat warval = W.inv() * tmpcor;
		
		if( (warval.at<float>(0) > mov.cols) || (warval.at<float>(0) < 0) || (warval.at<float>(1) > mov.rows) || (warval.at<float>(1) < 0))
		{
		continue;
		}
			
		float m,n,a,b;
		n = warval.at<float>(0);
		m = warval.at<float>(1);
		
		a = n - std::floor(n);
		b = m - std::floor(m);
		
		warped_fram.at<Vec3b>(y,x)[c] =((1-a)*(1-b)*(mov.at<Vec3b>(floor(m),floor(n))[c] ) + (1-a)*(b)*(mov.at<Vec3b>(floor(m) + 1, floor(n))[c]) + (1-b)*(a)*(mov.at<Vec3b>(floor(m),floor(n)+1)[c]) + a*b*(mov.at<Vec3b>(floor(m)+1,floor(n)+1)[c]) );	
				
			
		}
		
	}
	
	}

	return warped_fram;
}

Mat register_fram(Mat refframe, Mat movingframe, vector<Point2f> &srcPf, int cnt )			// refframe = previous / static frame === movingframe = next / moving frame
{	
	Mat warp_matrix;
	Mat refgray,movgray;
	cvtColor(refframe, refgray,CV_BGR2GRAY);
	cvtColor(movingframe, movgray,CV_BGR2GRAY);
	 
	vector<Point2f> dstPf;
	vector<uchar> status;
	vector<float> err;
	
	if( cnt%20 == 0 )
	{
	goodFeaturesToTrack(refgray, srcPf, 200, 0.01, 30);
	}
	
	calcOpticalFlowPyrLK(refgray,movgray,srcPf, dstPf,status,err);
	
	Mat mvgpts; Mat srcpts;
	
	for(int i = 0;i<srcPf.size();i++){	if(status[i]){ Mat mvtp = Mat::zeros(1,2,CV_32F); mvtp.at<float>(0) = dstPf[i].x; mvtp.at<float>(1) = dstPf[i].y; mvgpts.push_back(mvtp); }};
	for(int i = 0;i<dstPf.size();i++){	if(status[i]){ Mat sctp = Mat::zeros(1,2,CV_32F); sctp.at<float>(0) = srcPf[i].x; sctp.at<float>(1) = srcPf[i].y; srcpts.push_back(sctp);	}};
	
	Mat warp_m;
	
	cout<<"\n mvgpts size::"<<mvgpts.size()<<"\n";
	
	if( mvgpts.cols==0 )
	{
	warp_m = Mat::eye(2,3,CV_32F);
	}
	else
	{
	warp_m = get_affine_matrix(mvgpts,srcpts);
	}
	
	Mat warped_frame;// = warp_frame(movingframe, warp_m);
	warp_m.convertTo(warp_m,CV_32F);
	
	warpAffine(movingframe,warped_frame,warp_m, movingframe.size());
	
	srcPf = dstPf;
	return warped_frame;

}

Mat register_fram_dynamic(Mat refframe, Mat movingframe,Mat & warp_m )			// refframe = previous / static frame === movingframe = next / moving frame
{	
	Mat refgray,movgray;
	cvtColor(refframe, refgray,CV_BGR2GRAY);
	cvtColor(movingframe, movgray,CV_BGR2GRAY);
	 
	vector<Point2f> dstPf;
	vector<uchar> status;
	vector<float> err;
	vector<Point2f> srcPf;
		
	goodFeaturesToTrack(refgray, srcPf, 200, 0.01, 30);
	if(!srcPf.size()){return movingframe; }
	
	calcOpticalFlowPyrLK(refgray,movgray,srcPf, dstPf,status,err);
	Mat mvgpts; Mat srcpts;
	
	for(int i = 0;i<srcPf.size();i++){	if(status[i]){ Mat mvtp = Mat::zeros(1,2,CV_32F); mvtp.at<float>(0) = dstPf[i].x; mvtp.at<float>(1) = dstPf[i].y; mvgpts.push_back(mvtp); }};
	for(int i = 0;i<dstPf.size();i++){	if(status[i]){ Mat sctp = Mat::zeros(1,2,CV_32F); sctp.at<float>(0) = srcPf[i].x; sctp.at<float>(1) = srcPf[i].y; srcpts.push_back(sctp);	}};
	
//	cout<<"\n mvgpts size::"<<mvgpts.size()<<"\n";
	
	if( mvgpts.cols==0 || srcpts.cols == 0 )
	{
	return movingframe;
	}
	else
	{
	warp_m = get_affine_matrix(mvgpts,srcpts);
	}
	
	Mat warped_frame;// = warp_frame(movingframe, warp_m);
	warp_m.convertTo(warp_m,CV_32F);
	warpAffine(movingframe,warped_frame,warp_m, refframe.size());
	
	return warped_frame;

}

Mat multibandblend(Mat pano,			// The main pano
				   Mat tmppano,			// The tmppano
				   Mat cum_warp_mask,	// The mosiac mask (does not include current frame shape)
				   Mat inst_mos_mask,	// The current frame mask
				   Mat pano_r,			// The pano mask with the common region removed
				   Mat newmask,			// The portion of the current image mask which does not fit in with the pano
				   Mat res_mask,		// The region common to both the image masks
				   Mat prev_mos_mask,	// The previous mosaic mask
				   int npyramids = 3)
{
	vector<Mat> GP1,GP2,LP1,LP2;
	vector<Mat> LP3;
	vector<Mat> pr,nm,rrm,pf;
	
	////////////////////
	/// Get gaussian pyramid from pano and tmppano here
	
		GP1.push_back(pano);			// First Pyramid created out of original sized images
		GP2.push_back(tmppano);
		pr.push_back(pano_r);
		nm.push_back(newmask);
		rrm.push_back(res_mask);
		pf.push_back(prev_mos_mask);
	
	for(int i = 1; i< npyramids; i++)
	{
		int fact = pow(2,i);
		Mat tmp, tmp1;
		pyrDown(GP1[i-1],tmp);
		GP1.push_back(tmp);
		
		pyrDown(GP2[i-1],tmp1);
		GP2.push_back(tmp1);
		
		Mat pr1,nm1,rrm1,pf1;
		resize(prev_mos_mask,pf1,prev_mos_mask.size() / fact);
		resize(pano_r,pr1,pano_r.size() / fact);
		resize(newmask,nm1,newmask.size() / fact);
		resize(res_mask,rrm1,res_mask.size() / fact);
		
		pr.push_back(pr1);
		nm.push_back(nm1);
		rrm.push_back(rrm1);
		pf.push_back(pf1);
	}
	////////////////////
	
	///////////////////
	//// Get laplacian pyramids here
	
	for(int i = 0; i < GP1.size() - 1; i++ )
	{
	Mat LPt; pyrUp(GP1[i+1],LPt);
	LPt = GP1[i] - LPt;

	LP1.push_back(LPt);	
	}
	LP1.push_back(GP1[GP1.size() - 1]);
	
	for(int i = 0; i < GP2.size() - 1; i++ )
	{
	Mat LPT; pyrUp(GP2[i+1],LPT);
	LPT = GP2[i] - LPT;
	LP2.push_back(LPT);
	}
	LP2.push_back(GP2[GP2.size() - 1]);
	///////////////////
	
	//////
	// Initialize the third laplacian pyramid here
	for(int i = 0; i< LP1.size(); i++)
	{
	Mat LPi;
	LPi = Mat::zeros(LP1[i].size(),LP1[i].type());
	LP3.push_back(LPi);	
	}
	//////
	
	//////////////////////////////////////////////
	// Copy LP1 and LP2 with corresponding masks
	for(int i = 0; i< LP1.size();i++)
	{
	LP1[i].copyTo(LP3[i],pr[i]);
	LP2[i].copyTo(LP3[i],nm[i]);
	
	Mat TMP1,TMP2;
	LP1[i].copyTo(TMP1,rrm[i]);	
	LP2[i].copyTo(TMP2,rrm[i]);
	TMP1 = (TMP1 + TMP2) / 2;
	TMP1.copyTo(LP3[i],rrm[i]);
		
	}
	//////////////////////////////////////////////
	
	//////////////////////////////////////////////
	/// Assimilate all the masks
	
	Mat fp; LP3[LP3.size() - 1].copyTo(fp);
	
	for(int i = LP3.size() - 2; i >= 0; i--)
	{
	pyrUp(fp,fp);			// Upsampling it to lower level  
	fp += LP3[i];			// Adding to lower level pyramid
	}
	//////////////////////////////////////////////
	return fp;
}

Mat multibandblend_apples_oranges
				  (Mat pano,			// The main pano
				   Mat tmppano,			// The tmppano
				   Mat pano_r,			// The pano mask with the common region removed  (Region 1)
				   Mat newmask,			// The portion of the current image mask which does not fit in with the pano
				   Mat res_mask,		// The region common to both the image masks
				   int npyramids = 3)
{
	vector<Mat> GP1,GP2,LP1,LP2;
	vector<Mat> LP3;
	vector<Mat> pr,nm,rrm,pf;
	
	////////////////////
	/// Get gaussian pyramid from pano and tmppano here
	Mat tempreg; dilate(pano_r,tempreg,Mat::ones(3,3,CV_32F));
	tempreg = res_mask - tempreg; 
	Mat reg3 = res_mask - tempreg;			// mean region		(Region 3)
	Mat reg2; bitwise_or(tempreg,newmask,reg2);		// (Region 2)
	
	
	imshow("Region 3",reg3);
	imshow("Region 2",reg2);
	waitKey();
	
		GP1.push_back(pano);			// First Pyramid created out of original sized images
		GP2.push_back(tmppano);
		pr.push_back(pano_r);			// Region 1
		nm.push_back(reg2);				// Region 2
		rrm.push_back(reg3);			// Region 3
	
	for(int i = 1; i< npyramids; i++)
	{
		int fact = pow(2,i);
		Mat tmp, tmp1;
		pyrDown(GP1[i-1],tmp);
		GP1.push_back(tmp);
		
		pyrDown(GP2[i-1],tmp1);
		GP2.push_back(tmp1);
		
		Mat pr1,nm1,rrm1,pf1;
//		resize(prev_mos_mask,pf1,prev_mos_mask.size() / fact);
		resize(pano_r,pr1,pano_r.size() / fact);
		resize(reg2,nm1,reg2.size() / fact);
		resize(reg3,rrm1,reg3.size() / fact);
		
		pr.push_back(pr1);
		nm.push_back(nm1);
		rrm.push_back(rrm1);
//		pf.push_back(pf1);
	}
	////////////////////
	
	///////////////////
	//// Get laplacian pyramids here
	
	for(int i = 0; i < GP1.size() - 1; i++ )
	{
	Mat LPt; pyrUp(GP1[i+1],LPt);
	LPt = GP1[i] - LPt;

	LP1.push_back(LPt);	
	}
	LP1.push_back(GP1[GP1.size() - 1]);
	
	for(int i = 0; i < GP2.size() - 1; i++ )
	{
	Mat LPT; pyrUp(GP2[i+1],LPT);
	LPT = GP2[i] - LPT;
	LP2.push_back(LPT);
	}
	LP2.push_back(GP2[GP2.size() - 1]);
	///////////////////
	
	//////
	// Initialize the third laplacian pyramid here
	for(int i = 0; i< LP1.size(); i++)
	{
	Mat LPi;
	LPi = Mat::zeros(LP1[i].size(),LP1[i].type());
	LP3.push_back(LPi);	
	}
	//////
	
	//////////////////////////////////////////////
	// Copy LP1 and LP2 with corresponding masks
	for(int i = 0; i< LP1.size();i++)
	{
	LP1[i].copyTo(LP3[i],pr[i]);
	LP2[i].copyTo(LP3[i],nm[i]);
	
	Mat TMP1,TMP2;
	TMP1 = Mat::zeros(reg2.size(), CV_32F);	
	TMP2 = Mat::zeros(reg2.size(), CV_32F);	
	LP1[i].copyTo(TMP1,rrm[i]);	
	LP2[i].copyTo(TMP2,rrm[i]);
	TMP1 = (TMP1 + TMP2) / 2;
	TMP1.copyTo(LP3[i],rrm[i]);
		
	}
	//////////////////////////////////////////////
	
	//////////////////////////////////////////////
	/// Assimilate all the masks
	
	Mat fp; LP3[LP3.size() - 1].copyTo(fp);
	
	for(int i = LP3.size() - 2; i >= 0; i--)
	{
	pyrUp(fp,fp);			// Upsampling it to lower level  
	fp += LP3[i];			// Adding to lower level pyramid
	}
	//////////////////////////////////////////////
	return fp;
}

int main()
{

VideoCapture cap;
cap.open("/home/ml/motion/TEST_DOL_VID/dol5.mp4");
// cap.open("/home/ml/motion/TEST_DOL_VID/dol13.mp4");
int curr = 0;
if(!cap.isOpened())
{cout<<"\nError in Opening file \n";}
	

	Ptr<MotionEstimatorRansacL2> est = makePtr<MotionEstimatorRansacL2>(MM_AFFINE);
//	Ptr<TranslationBasedOutlierRejector> tblor = makePtr<TranslationBasedOutlierRejector>();
	Ptr<KeypointBasedMotionEstimator> kbest = makePtr<KeypointBasedMotionEstimator>(est);
    kbest->setDetector(GFTTDetector::create(10000));
//	kbest->setMotionModel(MM_AFFINE);
    
	
	Ptr<BackgroundSubtractorMOG2> B1 = createBackgroundSubtractorMOG2();
	B1->setHistory(10);
	B1->setNMixtures(7);
	vector<Mat> framl;
	
	Mat frame,bgmask,BG,BGold;
	/*
	cap>>frame;
	B1->apply(frame, bgmask);
	B1->getBackgroundImage(BG);
	medianBlur(BG,BG,7);
	framl.push_back(BG);
	
	cap>>frame;
	B1->apply(frame, bgmask);
	B1->getBackgroundImage(BG);
	medianBlur(BG,BG,7);
	framl.push_back(BG);
	*/
//	videostab::ImageMotionEstimatorBase;
	int iter_no = 775;
	double ssdhthresh = 600000;
	
	for(int j = 0; j < 20; j++ )			// Training for 20 frames
	{
	cap>>frame;  resize(frame, frame, Size(640,360));
	B1->apply(frame, bgmask);	
	B1->getBackgroundImage(BG);
//	medianBlur(BG,BG,7);
//	framl.push_back(BG);
	}							// Training BG subtractor
	cout<<"\n Training Complete!! \n";
	
	while(cap.get(CV_CAP_PROP_POS_FRAMES) < cap.get(CV_CAP_PROP_FRAME_COUNT))
	{
	cap>>frame;  resize(frame, frame, Size(640,360));
	B1->apply(frame, bgmask);	
	B1->getBackgroundImage(BGold);
	medianBlur(BGold,BGold,7);

		/*
	Mat _temone[3],tem1;
	_temone[0] = Mat::ones(frame.size(),CV_8UC1) * 255; 
	_temone[1] = Mat::ones(frame.size(),CV_8UC1) * 255; 
	_temone[2] = Mat::ones(frame.size(),CV_8UC1) * 255; 
	merge(_temone,tem1);
	imshow("temone",tem1);
		*/
	vector <Mat> matvect;
	matvect.push_back(frame);
	vector <Mat> BGmat;

	for (int i = 0; i < iter_no; i++ )
	{
		
		if( !(cap.get(CV_CAP_PROP_POS_FRAMES) < cap.get(CV_CAP_PROP_FRAME_COUNT)) )
		{
		break;
		}
	
	Mat fram; cap>>fram; resize(fram,fram,Size(640,360)); B1->apply(fram, bgmask); B1->getBackgroundImage(BG); 
	medianBlur(BG,BG,7);
	matvect.push_back(fram);
	BGmat.push_back(fram);					// Testing purposes only, obtaining mosaic from normal videos
											// Do BG later once idea fixed
//	BGmat.push_back(BG);	
	}
		
	clock_t st1 = clock();
	float startX = ( 3 * frame.cols ) / 2; float startY = ( 3 * frame.rows) / 2;			// Will decide the start and end region
		
	Mat pano = Mat::zeros(frame.rows*4,frame.cols*4,frame.type());
	Mat rsz; resize(BGmat[0],rsz,Size(frame.cols/2,frame.rows/2));
//	rsz.copyTo( pano(Rect(startX,startY,frame.cols / 2,frame.rows / 2 )) );
	double alpha = 0.5;
	Mat cum_w = Mat::eye(3,3,CV_32F);
	Mat rw = Mat::zeros(1,3,CV_32F);
	rw.at<float>(2) = 1;
	
	Mat tmp_warp = Mat::eye(3,3,CV_32F);			// initial warp for chaining warp
	tmp_warp.at<float>(0,2) = startX;
	tmp_warp.at<float>(1,2) = startY;
	Mat cum_warp_mask = Mat::zeros(pano.size(),CV_8UC1);
	Mat whit = Mat::ones(rsz.size(),CV_8UC1) * 255;	
	Mat prev_mos_mask;
		
		warpPerspective(rsz,pano,tmp_warp,pano.size());			// rsz moved onto the panorama palette by the means of a warp
		warpPerspective(whit,cum_warp_mask,tmp_warp,pano.size());			// rsz moved onto the panorama palette by the means of a warp
		cum_warp_mask.copyTo(prev_mos_mask);
		
		for(int ii = 1; ii < BGmat.size(); ii++)
		{
		/*	
		Stitcher stit = Stitcher::createDefault();
		PlaneWarper *warp = new PlaneWarper();

		Stitcher::Status stat = stit.stitch(tmpmat, pano);
		
		if(stat != Stitcher::OK)
		{
			
		cout<<"\nMosaic Generation Failed!!! Exit code:::"<<int(stat) <<" and BG vector size::"<<BGmat.size()<<"\n";
		exit(1);
		}			
		*/	
			
		Mat ffr;  BGmat[ii].copyTo(ffr);
		Mat teef,rfra;	
		resize(ffr,ffr,Size(frame.cols / 2,frame.rows / 2));
		Mat ff1; BGmat[ii - 1].copyTo(ff1); resize(ff1,ff1,Size( frame.cols / 2 , frame.rows / 2 ));	
		Mat inst_mos_mask = Mat::zeros(pano.size(),CV_8UC1);
			
			
//		regstECC(pano,ffr,rfra,MOTION_HOMOGRAPHY);
		rfra = register_fram_dynamic(ff1,ffr,teef);
		Mat tmppano = Mat::zeros(pano.size(),pano.type());
//		ffr.copyTo( tmppano(Rect(frame.cols / 4,frame.rows * 3,frame.cols / 2,frame.rows / 2 ))  );
		//	warpAffine(tmp,padded_window,tform, pano.size());
		teef.push_back(rw);
		tmp_warp *= teef;			// initial translation made inclusive of the chain
		cum_w *= teef;				// cumulative transform has the inbetween transform only (for the ROI)
			
			
		warpPerspective(ffr,tmppano,tmp_warp,tmppano.size(),INTER_LINEAR); 
		warpPerspective(whit,inst_mos_mask,tmp_warp,tmppano.size()); 
		tmppano.copyTo(pano,inst_mos_mask);								// one iteration moving forward
//// above will be done by pyamid accurately			
		
//		Rect min_rect = boundingRect(pntts);
//		tmpmm.release();
	
//			blender.prepare(min_rect);		/// may only be needed once
			
		Mat pano_r = cum_warp_mask - inst_mos_mask;
		Mat res_mask = cum_warp_mask - pano_r;
		Mat newmask = inst_mos_mask - res_mask;
			
			
pano =  multibandblend_apples_oranges(pano,tmppano,pano_r,newmask,res_mask,4);	
		
			
	/*	
		Mat idx,meanxy;
		findNonZero(res_mask,idx);			
		int Xsum = 0, Ysum = 0;
		for(int i = 0; i< idx.rows; i++)
		{
			int X,Y;
			Xsum += idx.at<Point>(i).x;
			Ysum += idx.at<Point>(i).y;
				
		}	
		Xsum = Xsum / idx.rows;
		Ysum = Ysum / idx.rows;	
		Point cen = Point(Xsum,Ysum);	
		clock_t clo1 = clock();	
		seamlessClone(tmppano,pano,res_mask,cen,pano,MIXED_CLONE);	
		clock_t clo2 = clock();
		cout<<"\n Time Taken for one seamless cloning::"<<(clo2 - clo1) / CLOCKS_PER_SEC<<"\n";	
			
			*/			// Seamless cloning::not applicable
	/*		
		Mat blen1,blen2,blen3;
		tmppano.copyTo(blen1,res_mask);
		pano.copyTo(blen2,res_mask);
			
		Mat YY; resize(tmppano,YY,Size(640,360));
	*		
//		imshow("Warped Pano",YY);
//		waitKey();	
		// first Mat can be ffr and second can be pano(Rect:: original position ) because in the end images moved with reference to original image, warping it to that point applies any deviation from that point automatically
		
			
			
		addWeighted(blen1,alpha,blen2,1-alpha,0,blen1);
	
		blen1.copyTo(pano,res_mask);
	*/
			
		bitwise_or(cum_warp_mask,inst_mos_mask,cum_warp_mask);
		inst_mos_mask.copyTo(prev_mos_mask);
//			pano = pano + tmppano;
		}
		/*
		Mat tmpP;
		clock_t ss1 = clock();
		blender.blend(pano,tmpP);
		clock_t ss2 = clock();
		
		cout<<"\n Time taken to blend::"<<(ss2 - ss1)/CLOCKS_PER_SEC<<"\n";
		*/
		clock_t st2 = clock();
		////////
		/// Testing only remove later
		cout<<"\n Time Taken for the panorama:::"<< (st2 - st1)/ CLOCKS_PER_SEC<<" \n";
		Mat tmo;
		cout<<"\n BGmat size:::"<<BGmat.size()<<"\n";
		resize(pano,pano,Size(640,360));
		imshow("panorama_new_blend_LARGE.jpeg",pano);
		waitKey();
		waitKey();
		exit(1);
		///////
		Mat tty = Mat::zeros(1,3,CV_32F);		
		tty.at<float>(2) = 1;
		
////////////
		// From this portion onwards only a pano image of BG's and original frames that made it needed
		
		for(int i = 0; i< matvect.size();i++)
	{
		Mat tform;
		Mat registered_frame = register_fram_dynamic(pano,matvect[i],tform);
		imshow("registered_frame",registered_frame);
		Mat diffimg = pano - registered_frame;										// diffimg has size of pano
		
//		imshow("Difference Image",diffimg);
			
//		Mat new_crop_paras = tform * crop_paras;
		Mat padded_window;					Mat	tmp = Mat::ones(matvect[i].size(),matvect[0].type()) * 255;
//		copyMakeBorder(Mat::zeros(matvect[i].size(), matvect[i].type),padded_window, (pano.rows - matvect[i].rows)/2,(pano.rows - matvect[i].rows)/2,(pano.cols - matvect[i].cols)/2,(pano.cols - matvect[i].cols)/2,BORDER_CONSTANT,0 );
		warpAffine(tmp,padded_window,tform, pano.size());
		
		imshow("Padded Window before use",padded_window);
			
		multiply(padded_window, diffimg,diffimg);									// Should take care of automatic multiplication of streams
		
		imshow("diffimg after multiplication",diffimg);

		Mat Y;
		tform.push_back( tty );
		tform  = tform.inv();
		Mat newform;
		newform.push_back(tform.row(0));
		newform.push_back(tform.row(1));

		warpAffine(diffimg,diffimg,newform, matvect[i].size(),INTER_LINEAR);
	    Mat gradif; cvtColor(diffimg,gradif,CV_BGR2GRAY );
		threshold(gradif, Y,100,1,THRESH_OTSU | THRESH_BINARY );	
//		Mat fin_res = diffimg(Rect((pano.rows - matvect[i].rows)/2,(pano.cols - matvect[i].cols)/2,frame.cols,frame.rows));
			// For testing only
		imshow("Final_Mask",Y*255);
		waitKey();
		exit(1);	
			//	waitKey(1);
	}
		
								// Mask operations with thresholding go here

								
		
	}
							
		/*
	Mat panorama; 	Mat diff,mask,tdiff;

	Stitcher sti = Stitcher::createDefault();			// would it make a difference if it wasnt declared every iteration?
	PlaneWarper * warper = new PlaneWarper;
	sti.setWarper(warper);
	cout<<"\n framl.size()"<<framl.size()<<"\n";
	if(framl.size() >  20)
	{
	Stitcher::Status stat = sti.stitch(framl,panorama);
	if(stat != Stitcher::OK){cout<<"\nError!! Mosaicing not done!!! \n"; waitKey();}
	imshow("Panorama",panorama);
	waitKey();
	}
	
								*/
//	resize(panorama,panorama,Size(1280,720));
//	Mat rggst = Mat::zeros(panorama.size(),panorama.type());	
//	frame.copyTo(rggst(Rect ((panorama.cols-frame.cols)/2 - 1,(panorama.rows - frame.rows)/2 - 1,frame.cols,frame.rows ) ));
//	regstECC(panorama,rggst,rggst,MOTION_HOMOGRAPHY);
//	Mat warp1 = kbest->estimate(rggst,panorama);
//	warpPerspective(rggst,rggst,warp1,panorama.size(),INTER_LINEAR);	
//	waitKey();
//	diff = panorama - rggst;
//	mask = diff > 100;
		
//	cout<<"\n Framl Size:::"<< framl.size() << " \n";
//		if(panorana){}
		

//	framl.erase( framl.begin() );		// A pano of three background frames created for which first frame deleted from dynamic framelist

return 1;
}