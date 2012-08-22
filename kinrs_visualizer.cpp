/* \author Geoffrey Biggs */


#include <iostream>
#include <vector>

#include <boost/thread/thread.hpp>
#include <pcl/common/common_headers.h>
#include <pcl/common/common_headers.h>
#include <pcl/features/normal_3d.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/vtk_lib_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/console/parse.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/registration/icp.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_plane.h>

typedef pcl::PointCloud<pcl::PointXYZ> cloud_t;
typedef pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_ptr_t;

pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_1_ptr;
pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_2_ptr;

pcl::PointCloud<pcl::PointXYZ>::Ptr reg_cloud_1_ptr(new pcl::PointCloud<pcl::PointXYZ>);
pcl::PointCloud<pcl::PointXYZ>::Ptr reg_cloud_2_ptr(new pcl::PointCloud<pcl::PointXYZ>);

pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_1_transformed_ptr(new pcl::PointCloud<pcl::PointXYZ>);

pcl::KdTreeFLANN<pcl::PointXYZ> kdtree_1;
pcl::KdTreeFLANN<pcl::PointXYZ> kdtree_2;

int reg_steps = 0;
const int num_reg_steps = 3; // 3
Eigen::Matrix4f reg_transform;

boost::shared_ptr<pcl::visualization::PCLVisualizer> visualizer;

enum {NONE, EXTRACT_PLANE, ICP} pick_mode = NONE;

void
printUsage (const char* progName)
{
  std::cout << "\n\nUsage: "<<progName<<" [options]\n\n"
            << "\n\n";
}


void
searchPointsAndAddToRegCloud(float x, float y, float z, pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud, pcl::KdTreeFLANN<pcl::PointXYZ>& kdtree, pcl::PointCloud<pcl::PointXYZ>::Ptr reg_cloud, float radius=0.1)
{
	pcl::PointXYZ searchPoint;
	searchPoint.x = x;
	searchPoint.y = y;
	searchPoint.z = z;
	std::vector<int> indiceList;
	std::vector<float> distanceList;

	kdtree.radiusSearch(searchPoint, radius, indiceList, distanceList);
	std::cout << "Found " << indiceList.size() << " points" << std::endl;

	for (int i=0; i < indiceList.size(); ++i) {
		reg_cloud->push_back(cloud->at(indiceList[i]));
	}
}

void
registration(pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud1, pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud2)
{
	pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
	icp.setInputCloud(cloud1);
	icp.setInputTarget(cloud2);

	pcl::PointCloud<pcl::PointXYZ> alignedCloud;
	icp.align(alignedCloud);

	std::cout << "has converged:" << icp.hasConverged() << " score: "
			<< icp.getFitnessScore() << std::endl;
	std::cout << icp.getFinalTransformation() << std::endl;

	reg_transform = icp.getFinalTransformation();

	pcl::transformPointCloud(*cloud_1_ptr, *cloud_1_transformed_ptr, reg_transform);

	std::cout << "Size of input cloud: " << cloud1->size() << " size of output cloud: " << cloud_1_transformed_ptr->size() << std::endl;
	std::cout << "Showing transformed cloud instead of Cloud #1" << std::endl;

	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> pc1tr_color(cloud_1_transformed_ptr, 255, 255, 0); // yellow
	visualizer->addPointCloud<pcl::PointXYZ> (cloud_1_transformed_ptr, pc1tr_color, "Cloud #1 transformed");

	visualizer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "Cloud #1 transformed");
	visualizer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_OPACITY, 0.5, "Cloud #1 transformed");
	//visualizer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_OPACITY, 0.0, "Cloud #1");
}

void
addToRegistration(float x, float y, float z)
{

	if (reg_steps < num_reg_steps) {
		searchPointsAndAddToRegCloud(x, y, z, cloud_1_ptr, kdtree_1, reg_cloud_1_ptr);
		searchPointsAndAddToRegCloud(x, y, z, cloud_2_ptr, kdtree_2, reg_cloud_2_ptr);

		reg_steps++;

		if (reg_steps == num_reg_steps) {
			std::cout << "Performing registration" << std::endl;
			std::cout << "Cloud #1 (reg): " << reg_cloud_1_ptr->size() << std::endl;
			std::cout << "Cloud #2 (reg): " << reg_cloud_2_ptr->size() << std::endl;

			visualizer->updatePointCloud(reg_cloud_1_ptr, "RegCloud #1");
			visualizer->updatePointCloud(reg_cloud_2_ptr, "RegCloud #2");

			registration(reg_cloud_1_ptr, reg_cloud_2_ptr);
		}
	}
}


/*void
ransacPlane(pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud)
{
	pcl::SampleConsensusModelPlane<pcl::PointXYZ> planeModel(cloud);
	pcl::RandomSampleConsensus<pcl::PointXYZ> ransac(planeModel);
	ransac.setDistanceThreshold(0.01);
	ransac.computeModel();
	Eigen::VectorXf plane_coeff;
	ransac.getModelCoefficients(plane_coeff);
	std::cout << "Plane coeffs: " << plane_coeff << std::endl;
}

void
alignGroundPlanes(float x, float y, float z)
{

	float radius = 0.15;
	searchPointsAndAddToRegCloud(x, y, z, cloud_1_ptr, kdtree_1, reg_cloud_1_ptr, radius);
	searchPointsAndAddToRegCloud(x, y, z, cloud_2_ptr, kdtree_2, reg_cloud_2_ptr, radius);

	ransacPlane(reg_cloud_1_ptr);
	ransacPlane(reg_cloud_2_ptr);

	reg_cloud_1_ptr->clear();
	reg_cloud_2_ptr->clear();
}*/

void toggleCloud(const char* cloud_id)
{
	double opacity;
	visualizer->getPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_OPACITY, opacity, cloud_id);
	visualizer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_OPACITY, opacity > 0 ? 0.0 : 0.5, cloud_id);
}

void keyboardCallback(const pcl::visualization::KeyboardEvent& event)
{
	char keycode = event.getKeyCode();

	double opacity;

	if (event.keyUp()) {
		switch (keycode) {
		case 't':
			if (reg_steps > 0) {
				reg_steps = 0;
				reg_cloud_1_ptr->clear();
				reg_cloud_2_ptr->clear();
				std::cout << "Resetting previously picked points" << std::endl;
			}
			pick_mode = ICP;
			break;
		case 'j':
			std::cout << "Registration using all points ICP" << std::endl;
			registration(cloud_1_ptr, cloud_2_ptr);
			break;
		case 'p':
			pick_mode = EXTRACT_PLANE;
			std::cout << "RANSAC plane extraction mode" << std::endl;
			break;
		case '1':
			toggleCloud("Cloud #1");
			break;
		case '2':
			toggleCloud("Cloud #2");
			break;
		case '3':
			toggleCloud("Cloud #1 transformed");
			break;
		case '4':
			toggleCloud("RegCloud #1");
			break;
		case '5':
			toggleCloud("RegCloud #2");
			break;
		default:
			break;
		}
	}

}

void
pointPickCallback(const pcl::visualization::PointPickingEvent& event)
{
	float x,y,z;

	if (pick_mode == NONE)
		return;

	event.getPoint(x, y, z);
	std::cout << "Picked point (" << x << ", " << y << ", " << z << ")" << std::endl;

	switch (pick_mode) {
	case ICP:
		addToRegistration(x, y, z);
		break;
	case EXTRACT_PLANE:
		//alignGroundPlanes(x, y, z);
		break;
	}

}

boost::shared_ptr<pcl::visualization::PCLVisualizer> dualPointCloudVis (pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud_1, pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud_2)
{
  // --------------------------------------------
  // -----Open 3D viewer and add point cloud-----
  // --------------------------------------------
  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("Dual viewer"));
  viewer->setBackgroundColor (0, 0, 0);
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> pc1_color(cloud_1, 0, 255, 0); // green
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> pc2_color(cloud_2, 255, 0, 0); // red
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> pc1reg_color(reg_cloud_1_ptr, 0, 100, 0); // darker green
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> pc2reg_color(reg_cloud_2_ptr, 100, 0, 0); // darker red

  viewer->addPointCloud<pcl::PointXYZ> (cloud_1, pc1_color, "Cloud #1");
  viewer->addPointCloud<pcl::PointXYZ> (cloud_2, pc2_color, "Cloud #2");
  viewer->addPointCloud<pcl::PointXYZ> (reg_cloud_1_ptr, pc1reg_color, "RegCloud #1");
  viewer->addPointCloud<pcl::PointXYZ> (reg_cloud_2_ptr, pc2reg_color, "RegCloud #2");

  viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "Cloud #1");
  viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "Cloud #2");
  viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "RegCloud #1");
  viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "RegCloud #2");

  viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_OPACITY, 0.5, "Cloud #1");
  viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_OPACITY, 0.5, "Cloud #2");
  viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_OPACITY, 0.5, "RegCloud #1");
  viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_OPACITY, 0.5, "RegCloud #2");

  viewer->registerKeyboardCallback(keyboardCallback);
  viewer->registerPointPickingCallback(pointPickCallback);

  viewer->addCoordinateSystem (1.0);
  viewer->initCameraParameters ();
  return (viewer);
}

boost::shared_ptr<pcl::visualization::PCLVisualizer> dualMeshVis (pcl::PolygonMesh::ConstPtr mesh_1, pcl::PolygonMesh::ConstPtr mesh_2)
{
  // --------------------------------------------
  // -----Open 3D viewer and add point cloud-----
  // --------------------------------------------
  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("Dual Mesh viewer"));
  viewer->setBackgroundColor (0, 0, 0);

  viewer->addPolygonMesh(*mesh_1, "Mesh #1");
  viewer->addPolygonMesh(*mesh_2, "Mesh #2");

  viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 255, 0, 0, "Mesh #1");
  viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0, 255, 0, "Mesh #2");

  viewer->addCoordinateSystem (1.0);
  viewer->initCameraParameters ();
  return (viewer);
}

pcl::PointCloud<pcl::PointXYZ>::Ptr
loadCloud(char* filename)
{
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

  if (pcl::io::loadPCDFile<pcl::PointXYZ> (filename, *cloud) == -1) //* load the file
  {
	PCL_ERROR ("Couldn't read cloud file %s\n", filename);
	return cloud;
  }
  std::cout << "Loaded "
			<< cloud->width * cloud->height
			<< " data points from " << filename
			<< std::endl;

  return cloud;
}

pcl::PolygonMesh::Ptr
loadMesh(char* filename)
{
  pcl::PolygonMesh::Ptr mesh(new pcl::PolygonMesh);

  if (pcl::io::loadPolygonFile(filename, *mesh) == -1) //* load the file
  {
	PCL_ERROR ("Couldn't read mesh file %s\n", filename);
	return mesh;
  }
  std::cout << "Loaded mesh from " << filename
			<< std::endl;

  return mesh;
}

// --------------
// -----Main-----
// --------------
int
main (int argc, char** argv)
{
  if (pcl::console::find_argument (argc, argv, "--mesh") >= 0) {
	  pcl::PolygonMesh::Ptr mesh_1_ptr = loadMesh(argv[1]);
	  pcl::PolygonMesh::Ptr mesh_2_ptr = loadMesh(argv[2]);

	  visualizer = dualMeshVis(mesh_1_ptr, mesh_2_ptr);
  }
  else {
	  // ------------------------------------
	  // ----- Load point clouds        -----
	  // ------------------------------------
	  cloud_1_ptr = loadCloud(argv[1]);
	  cloud_2_ptr = loadCloud(argv[2]);

	  // Empty registration point clouds
//	  reg_cloud_1_ptr = new pcl::PointCloud<pcl::PointXYZ>.makeShared();
//	  reg_cloud_2_ptr = new pcl::PointCloud<pcl::PointXYZ>.makeShared();

	  // Kd-trees of both clouds
	  kdtree_1.setInputCloud(cloud_1_ptr);
	  kdtree_2.setInputCloud(cloud_2_ptr);

	  //viewer = simpleVis(basic_cloud_ptr);
	  visualizer = dualPointCloudVis(cloud_1_ptr, cloud_2_ptr);
  }

  std::cout << argv[1] << " is green and " << argv[2] << " is red." << std::endl;

  //--------------------
  // -----Main loop-----
  //--------------------
  while (!visualizer->wasStopped ())
  {
    visualizer->spinOnce (100);
    boost::this_thread::sleep (boost::posix_time::microseconds (100000));
  }
}
