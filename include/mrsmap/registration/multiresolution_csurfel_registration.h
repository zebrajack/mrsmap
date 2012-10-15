/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2011, Computer Science Institute VI, University of Bonn
 *  Author: Joerg Stueckler, 16.05.2011
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of University of Bonn, Computer Science Institute 
 *     VI nor the names of its contributors may be used to endorse or 
 *     promote products derived from this software without specific 
 *     prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 */

#ifndef MULTIRESOLUTION_CSURFEL_REGISTRATION_H_
#define MULTIRESOLUTION_CSURFEL_REGISTRATION_H_

#include <gsl/gsl_multimin.h>

#include "mrsmap/map/multiresolution_csurfel_map.h"

#include "octreelib/algorithm/downsample.h"

#include <list>

// takes in two map for which it estimates the rigid transformation with a coarse-to-fine strategy.
namespace mrsmap {

class MultiResolutionColorSurfelRegistration {
public:

	class SurfelAssociation {
	public:
		SurfelAssociation()
				: n_src_( NULL ), src_( NULL ), src_idx_( 0 ), n_dst_( NULL ), dst_( NULL ), dst_idx_( 0 ), match( 0 ) {
		}
		SurfelAssociation( spatialaggregate::OcTreeNode< float, MultiResolutionColorSurfelMap::NodeValue >* n_src, MultiResolutionColorSurfelMap::Surfel* src, unsigned int src_idx,
				spatialaggregate::OcTreeNode< float, MultiResolutionColorSurfelMap::NodeValue >* n_dst, MultiResolutionColorSurfelMap::Surfel* dst, unsigned int dst_idx )
				: n_src_( n_src ), src_( src ), src_idx_( src_idx ), n_dst_( n_dst ), dst_( dst ), dst_idx_( dst_idx ), match( 1 ) {
		}
		~SurfelAssociation() {
		}

		spatialaggregate::OcTreeNode< float, MultiResolutionColorSurfelMap::NodeValue >* n_src_;
		MultiResolutionColorSurfelMap::Surfel* src_;
		unsigned int src_idx_;
		spatialaggregate::OcTreeNode< float, MultiResolutionColorSurfelMap::NodeValue >* n_dst_;
		MultiResolutionColorSurfelMap::Surfel* dst_;
		unsigned int dst_idx_;

		double df_tx;
		double df_ty;
		double df_tz;
		double df_qx;
		double df_qy;
		double df_qz;
		Eigen::Matrix< double, 6, 6 > d2f, JSzJ;
		double error;
		double weight;
		int match;

	public:
		EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	};

	typedef std::vector< SurfelAssociation, Eigen::aligned_allocator< SurfelAssociation > > SurfelAssociationList;

	struct RegistrationFunctionParameters {
		MultiResolutionColorSurfelMap* source;
		MultiResolutionColorSurfelMap* target;
		algorithm::OcTreeSamplingVectorMap< float, MultiResolutionColorSurfelMap::NodeValue >* targetSamplingMap;
		Eigen::Matrix4d* transform;
		float minResolution, maxResolution;
		float lastWSign;
		bool interpolate_neighbors;

		pcl::PointCloud< pcl::PointXYZ >::Ptr correspondences_source_points_;
		pcl::PointCloud< pcl::PointXYZ >::Ptr correspondences_target_points_;
	};

	void associateMapsBreadthFirst( SurfelAssociationList& surfelAssociations, MultiResolutionColorSurfelMap& source, MultiResolutionColorSurfelMap& target,
			algorithm::OcTreeSamplingVectorMap< float, MultiResolutionColorSurfelMap::NodeValue >& targetSamplingMap, Eigen::Matrix4d& transform, double minResolution, double maxResolution );
	void associateMapsBreadthFirstParallel( SurfelAssociationList& surfelAssociations, MultiResolutionColorSurfelMap& source, MultiResolutionColorSurfelMap& target,
			algorithm::OcTreeSamplingVectorMap< float, MultiResolutionColorSurfelMap::NodeValue >& targetSamplingMap, Eigen::Matrix4d& transform, double minResolution, double maxResolution,
			double searchDistFactor, double maxSearchDist, bool useFeatures );

	void associateNodeListParallel( SurfelAssociationList& surfelAssociations, MultiResolutionColorSurfelMap& source, MultiResolutionColorSurfelMap& target,
			std::vector< spatialaggregate::OcTreeNode< float, MultiResolutionColorSurfelMap::NodeValue >* >& nodes, int processDepth, Eigen::Matrix4d& transform, double searchDistFactor,
			double maxSearchDist, bool useFeatures );

	std::pair< int, int > calculateNegLogLikelihood( double& logLikelihood, spatialaggregate::OcTreeNode< float, MultiResolutionColorSurfelMap::NodeValue >* node_src,
			spatialaggregate::OcTreeNode< float, MultiResolutionColorSurfelMap::NodeValue >* node_tgt, const Eigen::Matrix4d& transform, bool interpolate );
	spatialaggregate::OcTreeNode< float, MultiResolutionColorSurfelMap::NodeValue >* calculateNegLogLikelihoodFeatureScoreN( double& logLikelihood, double& featureScore,
			spatialaggregate::OcTreeNode< float, MultiResolutionColorSurfelMap::NodeValue >* node, const MultiResolutionColorSurfelMap& target, const Eigen::Matrix4d& transform, bool interpolate =
					false );
	spatialaggregate::OcTreeNode< float, MultiResolutionColorSurfelMap::NodeValue >* calculateNegLogLikelihoodN( double& logLikelihood,
			spatialaggregate::OcTreeNode< float, MultiResolutionColorSurfelMap::NodeValue >* node, const MultiResolutionColorSurfelMap& target, const Eigen::Matrix4d& transform, bool interpolate =
					false );
	bool calculateNegLogLikelihood( double& likelihood, spatialaggregate::OcTreeNode< float, MultiResolutionColorSurfelMap::NodeValue >* node, const MultiResolutionColorSurfelMap& target,
			const Eigen::Matrix4d& transform, bool interpolate = false );

	double matchLogLikelihood( MultiResolutionColorSurfelMap& source, MultiResolutionColorSurfelMap& target, Eigen::Matrix4d& transform );

	bool estimateTransformationNewton( MultiResolutionColorSurfelMap& source, MultiResolutionColorSurfelMap& target, Eigen::Matrix4d& transform, float startResolution, float stopResolution,
			pcl::PointCloud< pcl::PointXYZ >::Ptr correspondencesSourcePoints, pcl::PointCloud< pcl::PointXYZ >::Ptr correspondencesTargetPoints, SurfelAssociationList* associations,
			int coarseToFineIterations, int fineIterations );
	bool estimateTransformationGradientDescent( MultiResolutionColorSurfelMap& source, MultiResolutionColorSurfelMap& target, Eigen::Matrix4d& transform, float startResolution, float stopResolution,
			pcl::PointCloud< pcl::PointXYZ >::Ptr correspondencesSourcePoints, pcl::PointCloud< pcl::PointXYZ >::Ptr correspondencesTargetPoints, SurfelAssociationList* associations,
			int maxIterations );

	bool estimateTransformation( MultiResolutionColorSurfelMap& source, const boost::shared_ptr< const pcl::PointCloud< pcl::PointXYZRGB > >& cloud,
			const boost::shared_ptr< const std::vector< int > >& indices, Eigen::Matrix4d& transform, float startResolution, float stopResolution,
			pcl::PointCloud< pcl::PointXYZ >::Ptr correspondencesSourcePoints, pcl::PointCloud< pcl::PointXYZ >::Ptr correspondencesTargetPoints, int gradientIterations = 25,
			int coarseToFineIterations = 5, int fineIterations = 5, SurfelAssociationList* associations = NULL );
	bool estimateTransformation( MultiResolutionColorSurfelMap& source, const pcl::PointCloud< pcl::PointXYZRGB >& cloud, const std::vector< int >& indices, Eigen::Matrix4d& transform,
			float startResolution, float stopResolution, pcl::PointCloud< pcl::PointXYZ >::Ptr correspondencesSourcePoints, pcl::PointCloud< pcl::PointXYZ >::Ptr correspondencesTargetPoints,
			int gradientIterations = 25, int coarseToFineIterations = 5, int fineIterations = 5, SurfelAssociationList* associations = NULL );
	bool estimateTransformation( MultiResolutionColorSurfelMap& source, MultiResolutionColorSurfelMap& target, Eigen::Matrix4d& transform, float startResolution, float stopResolution,
			pcl::PointCloud< pcl::PointXYZ >::Ptr correspondencesSourcePoints, pcl::PointCloud< pcl::PointXYZ >::Ptr correspondencesTargetPoints, int gradientIterations = 25,
			int coarseToFineIterations = 5, int fineIterations = 5, SurfelAssociationList* associations = NULL );

	bool estimatePoseCovariance( Eigen::Matrix< double, 6, 6 >& cov, MultiResolutionColorSurfelMap& source, MultiResolutionColorSurfelMap& target, Eigen::Matrix4d& transform, float startResolution,
			float stopResolution );

};

}


#endif /* MULTIRESOLUTION_SURFEL_REGISTRATION_H_ */

