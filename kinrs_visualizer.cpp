/* \author Geoffrey Biggs */


#include <iostream>
#include <vector>
#include <string>

#include <boost/thread/thread.hpp>
#include <pcl/common/common_headers.h>
//#include <pcl/features/normal_3d.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/vtk_lib_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/console/parse.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/registration/icp.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
//#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/features/normal_3d.h>
#include <pcl/filters/extract_indices.h>


typedef pcl::PointCloud<pcl::PointXYZ> cloud_t;
typedef pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_ptr_t;
typedef pcl::PolygonMesh mesh_t;
typedef pcl::PolygonMesh::Ptr mesh_ptr_t;
typedef pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_t;


class KinfuResultData {
public:
	KinfuResultData(std::string prefix);
	bool loadDirectory(std::string directory);
    cloud_ptr_t getCloud();
    std::string getPrefix();
    void transformMesh(Eigen::Matrix4f T);
    mesh_ptr_t getTransformedMesh();
    void setBaseColor(float r, float g, float b);
    pcl::RGB getBaseColor();
private:
    bool loadPointCloud(std::string directory);
    bool loadMesh(std::string directory);
    std::string m_prefix;
	cloud_ptr_t m_cloud;
	mesh_ptr_t m_mesh;
//	mesh_ptr_t m_transformed_mesh;
	std::string m_directory;
	pcl::RGB m_base_color;
};

KinfuResultData::KinfuResultData(std::string prefix) :
		m_prefix(prefix),
		m_cloud(new cloud_t),
		m_mesh(new mesh_t) {
	// Set default color to white
	m_base_color.r = m_base_color.g = m_base_color.b = 255;
}

void KinfuResultData::setBaseColor(float r, float g, float b) {
	m_base_color.r = (uint8_t) r * 255;
	m_base_color.g = (uint8_t) g * 255;
	m_base_color.b = (uint8_t) b * 255;
}

pcl::RGB KinfuResultData::getBaseColor() {
	return m_base_color;
}

cloud_ptr_t KinfuResultData::getCloud() {
	return m_cloud;
}

std::string KinfuResultData::getPrefix() {
	return m_prefix;
}

void KinfuResultData::transformMesh(Eigen::Matrix4f transform) {
	cloud_t tmp_cloud;
	cloud_t transformed_cloud;

	pcl::fromROSMsg(m_mesh->cloud, tmp_cloud);
	pcl::transformPointCloud(tmp_cloud, transformed_cloud, transform);
	pcl::toROSMsg(transformed_cloud, m_mesh->cloud);
}

mesh_ptr_t KinfuResultData::getTransformedMesh() {
	return m_mesh;
}

bool KinfuResultData::loadPointCloud(std::string directory) {
	std::string filename = directory + "/cloud_bin.pcd";
	std::cout << "Loading pointcloud from " << filename << std::endl;

	if (pcl::io::loadPCDFile<pcl::PointXYZ>(filename, *m_cloud) == -1) {
		PCL_ERROR("Couldn't read cloud file %s\n", filename.c_str());
		return false;
	}

	std::cout << "Cloud contained " << m_cloud->size() << " points." << std::endl;

	return true;
}

bool KinfuResultData::loadMesh(std::string directory) {
	std::string filename = directory + "/mesh.ply";

	std::cout << "Loading mesh from " << filename << std::cout;
	if (pcl::io::loadPolygonFilePLY(filename, *m_mesh) == -1) //* load the file
			{
		PCL_ERROR("Couldn't read mesh file %s\n", filename.c_str());
		return false;
	}
	std::cout << "Loaded mesh from " << filename << std::endl;

	return true;
}

bool KinfuResultData::loadDirectory(std::string directory) {
	loadPointCloud(directory);
	loadMesh(directory);
	return true;
}

enum DataFilter{
	DATA_ORIGINAL = 0,
	DATA_RECTIFIED = 1,
	DATA_ALL,
	DATA_NONE
};

class KinfuResultVisualizer {
public:
	KinfuResultVisualizer();
	bool loadData(std::string original_dir, std::string rectified_dir);
	void start(void);

private:
	// Methods
	void setup();
	bool addCloudFromData(KinfuResultData& data);
	bool addMeshFromData(KinfuResultData& data);
	void keyboardCallback(const pcl::visualization::KeyboardEvent& event, void* cookie);
	void mouseCallback(const pcl::visualization::MouseEvent& event, void*);
	void pointPickCallback(const pcl::visualization::PointPickingEvent& event, void*);
	void setPickEnabled(bool enabled);
	void alignMeshes(); // Requires points to have been picked first
	void estimatePlaneNormal(); // Requires picked points
	void drawLineNormal(); // Requires picked points
	void measureNormalDeviation(); // Requires picked points
    void measureBoxAngles(); // Requires picked points

	void toggleClouds(enum DataFilter cloud, bool enable);
	void toggleMeshes(enum DataFilter mesh, bool enable);

	// Variables
	bool m_pick_enabled;
	bool m_clouds_enabled[2];
	bool m_meshes_enabled[2];
	pcl::visualization::PCLVisualizer m_visualizer;
	KinfuResultData m_original_data;
	KinfuResultData m_rectified_data;
	std::vector<pcl::PointXYZ> m_registration_points;
	pcl::Normal m_ground_normal;
	bool m_has_plane;
	pcl::ModelCoefficients m_plane_coeff;
};

KinfuResultVisualizer::KinfuResultVisualizer() :
		m_visualizer("Visualizer"),
		m_original_data("Original"),
		m_rectified_data("Rectified"),
		m_pick_enabled(false),
		m_has_plane(false) {

	for (int i=0; i < 2; ++i) {
		m_clouds_enabled[i] = false;
		m_meshes_enabled[i] = false;
	}
	setPickEnabled(false);
}

void KinfuResultVisualizer::toggleClouds(enum DataFilter cloud, bool enable) {

	bool new_state[2] = {m_clouds_enabled[0], m_clouds_enabled[1]};

	switch (cloud) {
	case DATA_ORIGINAL:
	case DATA_RECTIFIED:
		new_state[cloud] = enable; // yes, the enable variable has a horrible name
		break;
	case DATA_ALL:
		for (int i=0; i < 2; ++i) {
			new_state[i] = enable;
		}
		break;
	default:
		return;
	}

	// Toggle
	for (int i=0; i < 2; ++i) {
		if (new_state[i] != m_clouds_enabled[i]) {
			bool enable = new_state[i];
			KinfuResultData& instance = (i == DATA_ORIGINAL) ? m_original_data : m_rectified_data;
			std::cout << (enable ? "Enabling" : "Disabling") <<
					" point cloud \"" << instance.getPrefix() << " cloud\"" << std::endl;
			if (enable) {
				addCloudFromData(instance);
			}
			else {
				m_visualizer.removePointCloud(instance.getPrefix() + " cloud");
			}
			m_clouds_enabled[i] = enable;
		}
	}
}

void KinfuResultVisualizer::toggleMeshes(DataFilter mesh, bool enable) {
	bool new_state[2] = {m_meshes_enabled[0], m_meshes_enabled[1]};

	switch (mesh) {
	case DATA_ORIGINAL:
	case DATA_RECTIFIED:
		new_state[mesh] = enable; // yes, the enable variable has a horrible name
		break;
	case DATA_ALL:
		for (int i=0; i < 2; ++i) {
			new_state[i] = enable;
		}
		break;
	default:
		return;
	}

	// Toggle
	for (int i=0; i < 2; ++i) {
		if (new_state[i] != m_meshes_enabled[i]) {
			bool enable = new_state[i];
			KinfuResultData& instance = (i == DATA_ORIGINAL) ? m_original_data : m_rectified_data;
			std::cout << (enable ? "Enabling" : "Disabling") <<
					" mesh \"" << instance.getPrefix() << " mesh\"" << std::endl;
			if (enable) {
				addMeshFromData(instance);
			}
			else {
				m_visualizer.removePolygonMesh(instance.getPrefix() + " mesh");
			}
			m_meshes_enabled[i] = enable;
		}
	}
}

bool KinfuResultVisualizer::loadData(std::string original_dir,
		std::string rectified_dir) {
	// Original data
	m_original_data.loadDirectory(original_dir);
	m_original_data.setBaseColor(0.0, 1.0, 0.0); // Green
	//this->addCloudFromData(m_original_data);

	// Rectified data
	m_rectified_data.loadDirectory(rectified_dir);
	m_rectified_data.setBaseColor(1.0, 0.0, 0.0); // Red
	//this->addCloudFromData(m_rectified_data);
}

bool KinfuResultVisualizer::addCloudFromData(KinfuResultData& data) {
	std::string cloud_name = data.getPrefix() + " cloud";
	std::cout << "Adding point cloud " << cloud_name << std::endl;
	pcl::RGB rgb_color = data.getBaseColor();
	m_visualizer.addPointCloud(data.getCloud(), cloud_name);
	m_visualizer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR,
			rgb_color.r / 255.0, rgb_color.g / 255.0, rgb_color.b / 255.0, cloud_name);
}

bool KinfuResultVisualizer::addMeshFromData(KinfuResultData& data) {
	std::string mesh_name = data.getPrefix() + " mesh";
	std::cout << "Adding mesh " << mesh_name << std::endl;
	pcl::RGB rgb_color = data.getBaseColor();

	m_visualizer.addPolygonMesh(*data.getTransformedMesh(), mesh_name);
	m_visualizer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR,
			rgb_color.r / 255.0, rgb_color.g / 255.0, rgb_color.b / 255.0, mesh_name);
	//m_visualizer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY,
	//			0.9, mesh_name);
}

void KinfuResultVisualizer::setup() {
	m_visualizer.registerKeyboardCallback(&KinfuResultVisualizer::keyboardCallback, *this);
	m_visualizer.registerPointPickingCallback(&KinfuResultVisualizer::pointPickCallback, *this);
	m_visualizer.registerMouseCallback(&KinfuResultVisualizer::mouseCallback, *this);
}

void KinfuResultVisualizer::start() {
	setup();
	toggleClouds(DATA_ALL, false);
	toggleMeshes(DATA_ALL, true);
	while (!m_visualizer.wasStopped()) {
		m_visualizer.spinOnce(100);
		boost::this_thread::sleep(boost::posix_time::microseconds(100000));
	}

}

void KinfuResultVisualizer::setPickEnabled(bool enabled) {
        if (m_pick_enabled == enabled) {
                return; // Do nothing
        }

        m_pick_enabled = enabled;

        if (m_pick_enabled) {
                // Reset previous picks
                m_registration_points.clear();
        }
        cout << "Picking: " << (m_pick_enabled ? "ON" : "OFF") << endl;
}


void KinfuResultVisualizer::keyboardCallback(const pcl::visualization::KeyboardEvent& event, void* cookie)
{
	char keycode = event.getKeyCode();

	if (event.keyUp()) {
		switch (keycode) {
		case 't':
			setPickEnabled(true);
			break;
		case 'a':
			alignMeshes();
			setPickEnabled(false); // Cleans up after registration
			break;
		case 'y':
			estimatePlaneNormal();
			setPickEnabled(false);
			break;
		case 'l':
			drawLineNormal();
			setPickEnabled(false);
			break;
		case 'k':
			measureBoxAngles();
			setPickEnabled(false);
			break;
			
		case 'b':
			m_visualizer.removeAllShapes();
			setPickEnabled(false);
			std::cout << "Clearing shapes" << std::cout;
			break;
		case '1':
			toggleClouds(DATA_ALL, ! (m_clouds_enabled[0] && m_clouds_enabled[1]));
			break;
		case '2':
			toggleMeshes(DATA_ALL, ! (m_meshes_enabled[0] && m_meshes_enabled[1]));
			break;
		case '6':	
		    if (m_meshes_enabled[DATA_ORIGINAL] == m_meshes_enabled[DATA_RECTIFIED])
		    {
		        toggleMeshes(DATA_ORIGINAL, ! m_meshes_enabled[DATA_ORIGINAL]);
		    }
		    else {
		        toggleMeshes(DATA_ORIGINAL, ! m_meshes_enabled[DATA_ORIGINAL]);
		        toggleMeshes(DATA_RECTIFIED, ! m_meshes_enabled[DATA_RECTIFIED]);
		    }		    
		    break;
		case '7':
			toggleClouds(DATA_ORIGINAL, ! m_clouds_enabled[DATA_ORIGINAL]);
			break;
		case '8':
			toggleClouds(DATA_RECTIFIED, ! m_clouds_enabled[DATA_RECTIFIED]);
			break;
		case '9':
			toggleMeshes(DATA_ORIGINAL, ! m_meshes_enabled[DATA_ORIGINAL]);
			break;
		case '0':
			toggleMeshes(DATA_RECTIFIED, ! m_meshes_enabled[DATA_RECTIFIED]);
			break;

		case 27: // ESC
			std::cout << "User exited application" << std::endl;
			m_visualizer.close();
		break;
		default:
			break;
		}
	}

}

void KinfuResultVisualizer::mouseCallback(const pcl::visualization::MouseEvent& event, void*)
{
    if (event.getType() != pcl::visualization::MouseEvent::MouseButtonRelease)
        return;
 
    if (!m_pick_enabled)
        return;
        
    if (!m_has_plane)
        return;    
        
    // Step 1: Get ray from camera to selected image coord
    std::vector<pcl::visualization::Camera> cameras;
    m_visualizer.getCameras(cameras);
    pcl::visualization::Camera camera = cameras[0];
    //vtk::vtkCamera vtkcam = camera.getCamera();
    cout << "Camera matrix" << endl;
    //cout << vtkcam.getRoll();
    Eigen::Matrix4d projmat;
    Eigen::Matrix4d viewmat;
    camera.computeProjectionMatrix(projmat);
    camera.computeViewMatrix(viewmat);
    
    Eigen::Matrix4d PV = projmat * viewmat;
    Eigen::Matrix4d PVinv = PV.inverse();
    
    Eigen::Vector4d img_point;
    img_point[0] = (2.0 * event.getX()) / camera.window_size[0] - 1.0;
    img_point[1] = (2.0 * event.getY()) / camera.window_size[1] - 1.0;
    img_point[2] = -1.0;
    img_point[3] = 1.0;
    
    Eigen::Vector4d ray_eye = projmat.inverse() * img_point;
    ray_eye[2] = -1.0;
    ray_eye[3] = 0.0;
    
    Eigen::Vector4d ray_wor = viewmat.inverse() * ray_eye;
    ray_wor.normalize();
    
    Eigen::Vector4d cam_pos(camera.pos[0], camera.pos[1], camera.pos[2], 1.0);
    
    Eigen::Vector4d other_point = cam_pos + 10.0 * ray_wor;
    
    //cout << "Homeogeneous image point " << img_point << endl;
    
    //Eigen::Vector4d world_point = PVinv*img_point;

    pcl::PointXYZ p1(other_point[0], other_point[1], other_point[2]);
    pcl::PointXYZ p2(cam_pos[0], cam_pos[1], cam_pos[2]);
    
    cout << "First point is " << p1 << "other point is " << p2 << endl;
    
    m_visualizer.addLine(p1, p2, "ray");
    
    // Step 2: Intersect with the plane
    Eigen::Vector3d ray_dir(ray_wor[0], ray_wor[1], ray_wor[2]);
    Eigen::Vector3d ray_start(cam_pos[0], cam_pos[1], cam_pos[2]);
    Eigen::Vector3d plane_normal(m_plane_coeff.values[0], m_plane_coeff.values[1], m_plane_coeff.values[2]);
    float d = m_plane_coeff.values[3];
    
    float t = - (d + plane_normal.dot(ray_start)) / plane_normal.dot(ray_dir);
    Eigen::Vector3d intersection = ray_start + ray_dir * t;
    pcl::PointXYZ point(intersection[0], intersection[1], intersection[2]);
    m_registration_points.push_back(point);
    cout << "Stored point" << point << endl;
}

void KinfuResultVisualizer::pointPickCallback(const pcl::visualization::PointPickingEvent& event, void*)
{
	float x,y,z;
	event.getPoint(x, y, z);
	std::cout << "Point = (" << x << ", " << y << ", " << z << ")" << std::endl;    
	
	if (!m_pick_enabled) {
		return;
	}

	std::cout << "Stored point for later use" << std::endl;
	pcl::PointXYZ point(x, y, z);
	m_registration_points.push_back(point);
}

void KinfuResultVisualizer::alignMeshes() {
	cloud_ptr_t orig_subcloud(new cloud_t);
	cloud_ptr_t rect_subcloud(new cloud_t);
	cloud_t alignedCloud;
	cloud_ptr_t orig_cloud;
	cloud_ptr_t rect_cloud;
	pcl::KdTreeFLANN<pcl::PointXYZ> orig_kdtree;
	pcl::KdTreeFLANN<pcl::PointXYZ> rect_kdtree;
	Eigen::Matrix4f T; // Transform

	double kdtree_search_radius = 0.1; // 1 dm

	if (m_registration_points.size() < 1) {
		std::cout << "No registration points selected. Can not align meshes." << std::endl;
		return;
	}

	std::cout << "Aligning clouds and meshes from " << m_registration_points.size() << " reference points" << std::endl;
	// Create new point clouds from radius from selected points
	orig_cloud = m_original_data.getCloud();
	rect_cloud = m_rectified_data.getCloud();
	orig_kdtree.setInputCloud(orig_cloud);
	rect_kdtree.setInputCloud(rect_cloud);
	std::vector<int> indiceList;
	std::vector<float> distanceList;
	for (int i=0; i < m_registration_points.size(); ++i) {
		pcl::PointXYZ searchPoint = m_registration_points[i];
		std::cout << "Point #" << i << ": " << searchPoint << std::endl;

		// Search original cloud
		indiceList.clear();
		distanceList.clear();
		orig_kdtree.radiusSearch(searchPoint, kdtree_search_radius, indiceList, distanceList);
		std::cout << "\t Found " << indiceList.size() << " points in original cloud." <<std::endl;
		for (int j=0; j < indiceList.size(); ++j) {
			orig_subcloud->push_back(orig_cloud->at(indiceList[j]));
		}

		// Cleanup and search rectified cloud
		indiceList.clear();
		distanceList.clear();
		rect_kdtree.radiusSearch(searchPoint, kdtree_search_radius, indiceList, distanceList);
		std::cout << "\t Found " << indiceList.size() << " points in rectified cloud." <<std::endl;
		for (int j=0; j < indiceList.size(); ++j) {
			rect_subcloud->push_back(rect_cloud->at(indiceList[j]));
		}
	}

	// ICP
	std::cout << "Running ICP. Orig subcloud: " << orig_subcloud->size() << " points, "
			<< "Rectified subcloud: " << rect_subcloud->size() << std::endl;

	pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
	icp.setInputCloud(orig_subcloud);
	icp.setInputTarget(rect_subcloud);
	icp.align(alignedCloud);

	T = icp.getFinalTransformation();

	std::cout << "Final transform is: " << std::endl << T << std::endl;

	// Transform original mesh
	m_original_data.transformMesh(T);

	// Turn off clouds, turn on meshes
	toggleClouds(DATA_ALL, false);
	toggleMeshes(DATA_ALL, true);
}

void KinfuResultVisualizer::estimatePlaneNormal() {

    if (m_meshes_enabled[DATA_ORIGINAL] == m_meshes_enabled[DATA_RECTIFIED]) {
		    std::cout << "Exactly ONE mesh must be active to do this operation" << std::endl;
		    return;
	    }

	    KinfuResultData& instance = m_meshes_enabled[DATA_ORIGINAL] ? m_original_data : m_rectified_data;

	cloud_ptr_t orig_subcloud(new cloud_t);
	cloud_ptr_t rect_subcloud(new cloud_t);
	cloud_t alignedCloud;
	cloud_ptr_t orig_cloud;
	cloud_ptr_t rect_cloud;
	pcl::KdTreeFLANN<pcl::PointXYZ> orig_kdtree;
	pcl::KdTreeFLANN<pcl::PointXYZ> rect_kdtree;
	Eigen::Matrix4f T; // Transform

	double kdtree_search_radius = 0.1; // 2cm for small box // 1 dm for large box

	if (m_registration_points.size() < 1) {
		std::cout << "No registration points selected. Can not align meshes." << std::endl;
		return;
	}

	std::cout << "Estimating plane from " << m_registration_points.size() << " reference points" << std::endl;
	// Get cloud from current mesh
	cloud_t tmp_cloud;
	pcl::fromROSMsg(instance.getTransformedMesh()->cloud, tmp_cloud);

	rect_kdtree.setInputCloud(tmp_cloud.makeShared());
	//std::vector<int> indiceList;
	pcl::PointIndices::Ptr indices(new pcl::PointIndices());
	for (int i=0; i < m_registration_points.size(); ++i) {
		pcl::PointXYZ searchPoint = m_registration_points[i];
		std::cout << "Point #" << i << ": " << searchPoint << std::endl;

		std::vector<int> _indiceList;
		std::vector<float> distanceList;
		rect_kdtree.radiusSearch(searchPoint, kdtree_search_radius, _indiceList, distanceList);
		//std::cout << "\t Found " << _indiceList.size() << " points in rectified cloud." <<std::endl;
		//indiceList.insert(indiceList.begin(), _indiceList.begin(), _indiceList.end());
		indices->indices.insert(indices->indices.begin(), _indiceList.begin(), _indiceList.end());
	}

    pcl::PointXYZ plane_center = m_registration_points[0];
	Eigen::Vector3f p(plane_center.x, plane_center.y, plane_center.z);
	
    m_plane_coeff.values.resize(4);
#ifndef USE_NORMAL_ESTIMATION
    cout << "Estimating plane using RANSAC" << endl;
	cloud_t ransac_cloud;
	pcl::ExtractIndices<pcl::PointXYZ> eifilter(true);
	eifilter.setInputCloud(tmp_cloud.makeShared());
	eifilter.setIndices(indices);
	eifilter.filter(ransac_cloud);
	
	pcl::SampleConsensusModelPlane<pcl::PointXYZ>::Ptr plane_model(new pcl::SampleConsensusModelPlane<pcl::PointXYZ>(ransac_cloud.makeShared()));
	pcl::RandomSampleConsensus<pcl::PointXYZ> ransac(plane_model);
	ransac.setDistanceThreshold(0.01);
	ransac.computeModel();
	Eigen::VectorXf ransac_plane_coeff;
	ransac.getModelCoefficients(ransac_plane_coeff);
    pcl::ModelCoefficients rpcf;
    m_plane_coeff.values[0] = ransac_plane_coeff[0];
    m_plane_coeff.values[1] = ransac_plane_coeff[1];
    m_plane_coeff.values[2] = ransac_plane_coeff[2];
    m_plane_coeff.values[3] = ransac_plane_coeff[3];
#else
    cout << "Estimating plane using normal estimation" << endl;
	pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normalEstimation;
	float nx, ny, nz, curvature;
	//std::cout << "Computing normal from " << indiceList.size() << " points." << std::endl;
	//normalEstimation.computePointNormal(tmp_cloud, indiceList, nx, ny, nz, curvature);
	normalEstimation.computePointNormal(tmp_cloud, indices->indices, nx, ny, nz, curvature);
	std::cout << "Plane normal: (" << nx << ", " << ny << ", " << nz << ") with curvature " << curvature << std::endl;

	m_ground_normal.normal_x = nx;
	m_ground_normal.normal_y = ny;
	m_ground_normal.normal_z = nz;
	
	Eigen::Vector3f normal(m_ground_normal.normal_x, m_ground_normal.normal_y, m_ground_normal.normal_z);
	normal.normalize();
	m_plane_coeff.values[0] = normal[0];
	m_plane_coeff.values[1] = normal[1];
	m_plane_coeff.values[2] = normal[2];
 	m_plane_coeff.values[3] = -normal.dot(p);
#endif
 	m_has_plane = true;
 	m_visualizer.addPlane(m_plane_coeff, p[0], p[1], p[2], "box_plane");
 	cout << "Got plane from " << instance.getPrefix() << endl;
}

void KinfuResultVisualizer::drawLineNormal() {
	if (m_registration_points.size() < 1) {
		return;
	}

	pcl::PointXYZ p = m_registration_points.back(); // Last point
	pcl::ModelCoefficients m;
	m.values.push_back(p.x);
	m.values.push_back(p.y);
	m.values.push_back(p.z);
	m.values.push_back(m_ground_normal.normal_x);
	m.values.push_back(m_ground_normal.normal_y);
	m.values.push_back(m_ground_normal.normal_z);

	m_visualizer.removeShape(std::string("normal"));
	m_visualizer.addLine(m, std::string("normal"));
	std::cout << "Adding line" << std::endl;
}

void KinfuResultVisualizer::measureNormalDeviation() {
	if (m_registration_points.size() < 6) {
		return;
	}

	// Points
	pcl::PointXYZ base_rect = m_registration_points[0];
	pcl::PointXYZ top_rect = m_registration_points[1];
	pcl::PointXYZ base_orig = m_registration_points[2];
	pcl::PointXYZ top_orig = m_registration_points[3];
	pcl::PointXYZ plane_1 = m_registration_points[4];
	pcl::PointXYZ plane_2 = m_registration_points[5];

	// Directions
	Eigen::Vector3f base_top_rect(top_rect.x - base_rect.x, top_rect.y - base_rect.y, top_rect.z - base_rect.z);
	Eigen::Vector3f base_top_orig(top_orig.x - base_orig.x, top_orig.y - base_orig.y, top_orig.z - base_orig.z);

	Eigen::Vector3f v1(plane_2.x - plane_1.x, plane_2.y - plane_1.y, plane_2.z - plane_1.z);
	Eigen::Vector3f normal(m_ground_normal.normal_x, m_ground_normal.normal_y, m_ground_normal.normal_z);

	// Project base_top_rect to lie in plane
	v1.normalize();
	normal.normalize();
	Eigen::Vector3f plane_normal = v1.cross(normal);
	base_top_rect = base_top_rect - plane_normal.dot(base_top_rect) * plane_normal;
	base_top_orig = base_top_orig - plane_normal.dot(base_top_orig) * plane_normal;


	// Calculate angle to normal
	base_top_rect.normalize();
	base_top_orig.normalize();
	float angle_rect_normal = acos(normal.dot(base_top_rect));
	float angle_orig_normal = acos(normal.dot(base_top_orig));
	float angle_rect_orig = acos(base_top_rect.dot(base_top_orig));

	std::cout << "Rectified, angle to normal: " << RAD2DEG(angle_rect_normal) << std::endl;
	std::cout << "Original, angle to normal: " << RAD2DEG(angle_orig_normal) << std::endl;
	std::cout << "Angle between rectified and original: " << RAD2DEG(angle_rect_orig) << std::endl;
	std::cout << "Normal: " << normal << std::endl;

	struct {
		pcl::PointXYZ p;
		Eigen::Vector3f n;
		std::string id;
	} lines[3] = {
			{base_rect, base_top_rect, "line_rect"},
			{base_orig, base_top_orig, "line_orig"},
			{plane_1, v1, "line_plane"}
	};

	for (int i=0; i < 3; ++i) {
		pcl::ModelCoefficients m;
		m.values.push_back(lines[i].p.x);
		m.values.push_back(lines[i].p.y);
		m.values.push_back(lines[i].p.z);
		m.values.push_back(lines[i].n[0]);
		m.values.push_back(lines[i].n[1]);
		m.values.push_back(lines[i].n[2]);

		m_visualizer.removeShape(lines[i].id);
		m_visualizer.addLine(m, lines[i].id);
		std::cout << "Adding line " << lines[i].id << std::endl;
		std::cout << "Coeffs: ";
		for (int j=0; j < m.values.size(); ++j)
			std::cout << m.values[j] << "  ";
		std::cout << std::endl;

	}

	m_visualizer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0.0, 0.0, 1.0, lines[0].id);
	m_visualizer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1.0, 0.0, 1.0, lines[1].id);
}

void PrintNumPy(Eigen::Vector3f& v, char* name) {
	std::cout << name << " = numpy.array(" << std::endl;
	std::cout << "[" << v[0] << ", " << v[1] << ", " << v[2] << "]" << std::endl;
	std::cout << ")" << std::endl;
}

void PrintNumPy(pcl::PointXYZ& v, char* name) {
	std::cout << name << " = numpy.array(" << std::endl;
	std::cout << "[" << v.x << ", " << v.y << ", " << v.z << "]" << std::endl;
	std::cout << ")" << std::endl;
}

void KinfuResultVisualizer::measureBoxAngles() {
	if (m_registration_points.size() != 4) {
		return;
	}

	if (m_meshes_enabled[DATA_ORIGINAL] == m_meshes_enabled[DATA_RECTIFIED]) {
		std::cout << "Exactly ONE mesh must be active to do this operation" << std::endl;
		return;
	}

	KinfuResultData& instance = m_meshes_enabled[DATA_ORIGINAL] ? m_original_data : m_rectified_data;

	// Points
	pcl::PointXYZ A = m_registration_points[0];
	pcl::PointXYZ B = m_registration_points[1];
	pcl::PointXYZ C = m_registration_points[2];
	pcl::PointXYZ D = m_registration_points[3];

    // Constrain to plane
    

	// Directions
	Eigen::Vector3f AB(B.x - A.x, B.y - A.y, B.z - A.z);
	Eigen::Vector3f BC(C.x - B.x, C.y - B.y, C.z - B.z);
	Eigen::Vector3f CD(D.x - C.x, D.y - C.y, D.z - C.z);
	Eigen::Vector3f DA(A.x - D.x, A.y - D.y, A.z - D.z);

	Eigen::Vector3f normal(m_ground_normal.normal_x, m_ground_normal.normal_y, m_ground_normal.normal_z);

	// Print vectors
	PrintNumPy(A, "A");
	PrintNumPy(B, "B");
	PrintNumPy(C, "C");
	PrintNumPy(D, "D");
	PrintNumPy(AB, "AB");
	PrintNumPy(BC, "BC");
	PrintNumPy(CD, "CD");
	PrintNumPy(DA, "DA");

	double angle_ABC = acos(AB.normalized().dot(BC.normalized()));
	double angle_BCD = acos(BC.normalized().dot(CD.normalized()));

	std::cout << "Angle ABC (deg): " << RAD2DEG(angle_ABC) << std::endl;
	std::cout << "Angle BCD (deg): " << RAD2DEG(angle_BCD) << std::endl;
	std::cout << "Length AB: " << AB.norm() << std::endl;
	std::cout << "Length BC: " << BC.norm() << std::endl;
	std::cout << "Length CD: " << CD.norm() << std::endl;
	std::cout << "Length DA: " << DA.norm() << std::endl;


    std::cout << "----- On one CSV line below -----" << std::endl;
    std::cout << instance.getPrefix() << ", ";
    std::cout <<  RAD2DEG(angle_ABC) << ", ";
	std::cout << RAD2DEG(angle_BCD) << ", ";
	std::cout << AB.norm() << ", ";
	std::cout << BC.norm() << ", ";
	std::cout << CD.norm() << ", ";
	std::cout << DA.norm() << std::endl;

	struct {
		pcl::PointXYZ p;
		Eigen::Vector3f n;
		std::string id;
	} lines[4] = {
			{A, AB, instance.getPrefix() + " AB"},
            {B, BC, instance.getPrefix() + " BC"},
			{C, CD, instance.getPrefix() + " CD"},
            {D, DA, instance.getPrefix() + " DA"},
	};

	pcl::RGB rgb = instance.getBaseColor();

	for (int i=0; i < 4; ++i) {
		pcl::ModelCoefficients m;
		m.values.push_back(lines[i].p.x);
		m.values.push_back(lines[i].p.y);
		m.values.push_back(lines[i].p.z);
		m.values.push_back(lines[i].n[0]);
		m.values.push_back(lines[i].n[1]);
		m.values.push_back(lines[i].n[2]);

		m_visualizer.removeShape(lines[i].id);
		m_visualizer.addLine(m, lines[i].id);
		std::cout << "Adding line " << lines[i].id << std::endl;
		std::cout << "Coeffs: ";
		for (int j=0; j < m.values.size(); ++j)
			std::cout << m.values[j] << "  ";
		std::cout << std::endl;
		m_visualizer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, rgb.r / 255.0, rgb.g / 255.0, rgb.b / 255.0, lines[i].id);

	}
}

// --------------
// -----Main-----
// --------------
int
main (int argc, char** argv)
{
	KinfuResultVisualizer visualizer;
	std::string orig_dir(argv[1]);
	std::string rect_dir;

	if (argc > 2) {
		rect_dir = argv[2];
	}
	else {
		rect_dir = orig_dir + "/result/";
	}

	visualizer.loadData(orig_dir, rect_dir);

	visualizer.start();
}


