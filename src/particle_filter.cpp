/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <cmath> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
    
    //cout<<"Enter ParticleFilter::init"<<endl;
    num_particles = 100;
    
    random_device           rdev{};
    default_random_engine   gen{rdev()};
    
    normal_distribution<double> x_ran(x, std[0]);
	normal_distribution<double> y_ran(y, std[1]);
	normal_distribution<double> theta_ran(theta, std[2]);
    
    particles.reserve(num_particles);
    
    for(int i=0; i< num_particles; i++) {
        Particle new_particle;
        new_particle.id = i;
        new_particle.x = x_ran(gen);
        new_particle.y = y_ran(gen);
        new_particle.theta = theta_ran(gen);
        new_particle.weight = 1.0;
        particles.push_back(new_particle);
    }
    
    /* debug purpose */
    //for (auto &one_part : particles) {
    //    cout<<"id:"<<one_part.id<< "x:"<<one_part.x<<" y:"<<one_part.y<<" theta:"<<one_part.theta<<endl;
    //}
    
    /* Resize vector weights in term of number of particles */
    weights.resize(num_particles);
    
    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

    //cout<<"enter ParticleFilter::prediction"<<endl;
    random_device           rdev{};
    default_random_engine   gen{rdev()};
    
    normal_distribution<double> x_ran(0, std_pos[0]);
	normal_distribution<double> y_ran(0, std_pos[1]);
    normal_distribution<double> theta_ran(0, std_pos[2]);
    
    for(auto &one:particles) {
        double x=one.x;
        double y=one.y;
        double theta = one.theta;

        if (yaw_rate > 1e-6) {
            one.x = x + velocity/yaw_rate * (sin(theta + yaw_rate*delta_t) - sin(theta));
            one.y = y + velocity/yaw_rate * (cos(theta) - cos(theta+yaw_rate*delta_t));
            one.theta = theta + yaw_rate * delta_t;
        } else {
            one.x = x + velocity * cos(theta) * delta_t;
            one.y = y + velocity * sin(theta) * delta_t;
        }
        one.x = one.x + x_ran(gen);
        one.y = one.y + y_ran(gen);
        one.theta = one.theta + theta_ran(gen);
    }
    
    //for (auto &one_part : particles) {
    //    cout<<"id:"<<one_part.id<<" x:"<<one_part.x<<" y:"<<one_part.y<<" theta:"<<one_part.theta<<endl;
    //}
}

void ParticleFilter::dataAssociation(Map& map_landmarks, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

    for (auto &obs: observations) {
        auto &pos  = map_landmarks.landmark_list[0];
        double dist1 = HUGE_VAL;
        for (auto &land_mark: map_landmarks.landmark_list) {
            double dist2 = dist(land_mark.x_f, land_mark.y_f, obs.x, obs.y);
            if (dist2 < dist1) {
                pos = land_mark;
                dist1 = dist2;
            }
        }
        obs.id = pos.id_i;
        
        //cout<<obs.id<<" "<<obs.x<<" "<<obs.y<<endl;
        //cout<<pos.id_i<<" "<<pos.x_f<<" "<<pos.y_f<<endl; 
    }
    //cout << endl;
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

    //cout<<"\nobservations:"<<observations.size()<<endl;
    //cout<<"map_landmarks:"<<map_landmarks.landmark_list.size()<<endl;
    //cout <<"weights:";
    int count = 0;
    for (auto & one_part:particles) {
    // for every particle
        // transform observation to map coordinate
        vector<LandmarkObs> obs_worldmap;
        obs_worldmap.reserve(observations.size());
        
        LandmarkObs obs;
        for (auto &obs1: observations){
            obs.x = one_part.x + obs1.x * cos(one_part.theta) - obs1.y * sin(one_part.theta);
            obs.y = one_part.y + obs1.x * sin(one_part.theta) + obs1.y * cos(one_part.theta);
            obs_worldmap.push_back(obs);
        }
        // associate observation with landmark
        dataAssociation(map_landmarks, obs_worldmap);
        
        // update weights the particle
        double w = 1.0;
        for (auto &obs1: obs_worldmap) {
            auto &pos = map_landmarks.landmark_list[obs1.id-1];
            
            //cout<<obs1.id<<" "<<obs1.x<<" "<<obs1.y<<endl;
            //cout<<pos.id_i<<" "<<pos.x_f<<" "<<pos.y_f<<endl;

            double exp_part = exp(-(pow((obs1.x - pos.x_f), 2)/2.0/pow(std_landmark[0], 2) \
                                + pow((obs1.y - pos.y_f), 2)/2.0/pow(std_landmark[1], 2)));

            double proba = 0.5/M_PI/std_landmark[0]/std_landmark[1]*exp_part;
            
            //cout << obs1.id <<":"<< proba <<endl;
            w = w*proba;
        }
        //cout << w <<" ";
        one_part.weight = w;
        weights[count++] = w;
    }
    //cout << endl;
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
    
    //cout <<"Resample:"<<endl;
    
    random_device           rdev{};

    default_random_engine   gen{rdev()};
    discrete_distribution<int>     d(weights.begin(), weights.end());
    
    vector<Particle> resampled;
    resampled.reserve(num_particles);
    
    int count = num_particles;
    while (count > 0) {
        int idx = d(gen);
        resampled.push_back(particles[idx]);
        count --;
        //cout <<idx<<" ";
    }
    //cout<<endl;
    particles = resampled;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
