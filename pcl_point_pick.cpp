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

#include <pcl/conversions.h>

typedef pcl::PointCloud<pcl::PointXYZ> cloud_t;
typedef pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_ptr_t;
typedef pcl::PolygonMesh mesh_t;
typedef pcl::PolygonMesh::Ptr mesh_ptr_t;
typedef pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_t;

class PointPicker {
public:
	PointPicker();
	bool loadMesh(std::string filename);
	void start(void);

private:
	// Methods
	void setup();
	void keyboardCallback(const pcl::visualization::KeyboardEvent& event, void* cookie);
	void mouseCallback(const pcl::visualization::MouseEvent& event, void*);
	void pointPickCallback(const pcl::visualization::PointPickingEvent& event, void*);
	void setPickEnabled(bool enabled);
	void estimatePlaneNormal(); // Requires picked points

	// Variables
	bool m_pick_enabled;
	mesh_t m_mesh;
	pcl::visualization::PCLVisualizer m_visualizer;
	std::vector<pcl::PointXYZ> m_registration_points;
	bool m_has_plane;
	pcl::ModelCoefficients m_plane_coeff;
};

PointPicker::PointPicker() :
		m_visualizer("Visualizer"),
		m_pick_enabled(false),
		m_has_plane(false) {
	setPickEnabled(false);
}

bool PointPicker::loadMesh(std::string filename) {
	std::cout << "Loading mesh from " << filename << std::cout;
	if (pcl::io::loadPolygonFilePLY(filename, m_mesh) == -1) //* load the file
			{
		PCL_ERROR("Couldn't read mesh file %s\n", filename.c_str());
		return false;
	}
	std::cout << "Loaded mesh from " << filename << std::endl;
	m_visualizer.addPolygonMesh(m_mesh, "mesh");
	return true;
}

void PointPicker::setup() {
	m_visualizer.registerKeyboardCallback(&PointPicker::keyboardCallback, *this);
	m_visualizer.registerPointPickingCallback(&PointPicker::pointPickCallback, *this);
	m_visualizer.registerMouseCallback(&PointPicker::mouseCallback, *this);
}

void PointPicker::start() {
	setup();
	while (!m_visualizer.wasStopped()) {
		m_visualizer.spinOnce(100);
		boost::this_thread::sleep(boost::posix_time::microseconds(100000));
	}

}

void PointPicker::setPickEnabled(bool enabled) {
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


void PointPicker::keyboardCallback(const pcl::visualization::KeyboardEvent& event, void* cookie)
{
	char keycode = event.getKeyCode();

	if (event.keyUp()) {
		switch (keycode) {
		case 't':
			setPickEnabled(true);
			break;
		case 'y':
			estimatePlaneNormal();
			setPickEnabled(false);
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

void PointPicker::mouseCallback(const pcl::visualization::MouseEvent& event, void*)
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

void PointPicker::pointPickCallback(const pcl::visualization::PointPickingEvent& event, void*)
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

void PointPicker::estimatePlaneNormal() {
	cloud_ptr_t cloud(new cloud_t);
	pcl::fromPCLPointCloud2(m_mesh.cloud, *cloud);
	pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
	Eigen::Matrix4f T; // Transform

	double kdtree_search_radius = 0.1; // 2cm for small box // 1 dm for large box

	if (m_registration_points.size() < 1) {
		std::cout << "No registration points selected. Can not align meshes." << std::endl;
		return;
	}

	std::cout << "Estimating plane from " << m_registration_points.size() << " reference points" << std::endl;
	// Get cloud from current mesh
	kdtree.setInputCloud(cloud);
	pcl::PointIndices::Ptr indices(new pcl::PointIndices());
	for (int i=0; i < m_registration_points.size(); ++i) {
		pcl::PointXYZ searchPoint = m_registration_points[i];
		std::cout << "Point #" << i << ": " << searchPoint << std::endl;

		std::vector<int> _indiceList;
		std::vector<float> distanceList;
		kdtree.radiusSearch(searchPoint, kdtree_search_radius, _indiceList, distanceList);
		std::cout << "\t Found " << _indiceList.size() << " points." <<std::endl;
		indices->indices.insert(indices->indices.begin(), _indiceList.begin(), _indiceList.end());
	}

    pcl::PointXYZ plane_center = m_registration_points[0];
    Eigen::Vector3f p(plane_center.x, plane_center.y, plane_center.z);
	
    m_plane_coeff.values.resize(4);

    cout << "Estimating plane using RANSAC" << endl;
	cloud_t ransac_cloud;
	pcl::ExtractIndices<pcl::PointXYZ> eifilter(true);
	eifilter.setInputCloud(cloud);
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

 	m_has_plane = true;
 	m_visualizer.addPlane(m_plane_coeff, p[0], p[1], p[2], "box_plane");
	
	std::cout << "Found plane " << m_plane_coeff.values[0] << ", ";
	std::cout << m_plane_coeff.values[1] << ", ";
	std::cout << m_plane_coeff.values[2] << ", ";
	std::cout << m_plane_coeff.values[3] << std::endlq;
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

// --------------
// -----Main-----
// --------------
int
main (int argc, char** argv)
{
	PointPicker picker;
	std::string mesh_path(argv[1]);

	picker.loadMesh(mesh_path);

	picker.start();
}


