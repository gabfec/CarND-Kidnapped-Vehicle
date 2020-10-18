/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>
#include <assert.h>

#include "helper_functions.h"

using std::string;
using std::vector;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  num_particles = 100;

  std::default_random_engine gen;

  std::normal_distribution<double> dist_x(x, std[0]);
  std::normal_distribution<double> dist_y(y, std[1]);
  std::normal_distribution<double> dist_theta(theta, std[2]);

  for (auto i = 0; i < num_particles; i++)
  {
    Particle particle;
    particle.id = i;
    particle.x = dist_x(gen);
    particle.y = dist_y(gen);
    particle.theta = dist_theta(gen);
    particle.weight = 1;

    particles.push_back(particle);
    weights.push_back(particle.weight);
  }

  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * Add measurements to each particle and add random Gaussian noise.
   */

  std::default_random_engine gen;

  for (auto& particle : particles)
  {

    if (yaw_rate > std::numeric_limits<double>::epsilon())
    {
      particle.x += velocity / yaw_rate * (sin(particle.theta + yaw_rate * delta_t) - sin(particle.theta));
      particle.y += velocity / yaw_rate * (cos(particle.theta) - cos(particle.theta + yaw_rate * delta_t));
      particle.theta += yaw_rate * delta_t;
    }
    else
    {
      particle.x += velocity * delta_t * cos(particle.theta);
      particle.y += velocity * delta_t * sin(particle.theta);
    }

    // Add random Gaussian noise
    std::normal_distribution<double> dist_x(particle.x, std_pos[0]);
    std::normal_distribution<double> dist_y(particle.y, std_pos[1]);
    std::normal_distribution<double> dist_theta(particle.theta, std_pos[2]);
    particle.x = dist_x(gen);
    particle.y = dist_y(gen);
    particle.theta = dist_theta(gen);
  }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   */

  for (auto& observation:  observations)
  {
    double min_distance = std::numeric_limits<double>::max();

    auto distance = [](LandmarkObs first, LandmarkObs second)
    {
      return sqrt(pow(first.x - second.x, 2) + pow(first.y - second.y, 2));
    };

    auto closest_landmark = -1;
    for (auto pred: predicted)
    {
      auto dist = distance(pred, observation);
      if (dist < min_distance)
      {
        min_distance = dist;
        closest_landmark = pred.id;
        //observation.id = closest_landmark;
      }
    }
    observation.id = closest_landmark;
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */

  auto multiv_prob = [] (double sig_x, double sig_y, double x_obs, double y_obs,
                   double mu_x, double mu_y)
   {
      // calculate normalization term
      auto gauss_norm = 1 / (2 * M_PI * sig_x * sig_y);

      // calculate exponent
      auto exponent = (pow(x_obs - mu_x, 2) / (2 * pow(sig_x, 2)))
                    + (pow(y_obs - mu_y, 2) / (2 * pow(sig_y, 2)));

      // calculate weight using normalization terms and exponent
      auto weight = gauss_norm * exp(-exponent);
      return weight;
   };

  weights.clear();

  for (auto particle: particles)
  {
    vector<LandmarkObs> translated_observations;

    for (auto observation: observations)
    {
      LandmarkObs trans_obs;

      trans_obs.x = particle.x + (observation.x * cos(particle.theta)) - (observation.y * sin(particle.theta));
      trans_obs.y = particle.y + (observation.x * sin(particle.theta)) + (observation.y * cos(particle.theta));
      trans_obs.id = observation.id;

      translated_observations.push_back(trans_obs);
    }

    std::vector<LandmarkObs> predicted_landmarks;
    for (auto landmark: map_landmarks.landmark_list)
    {
      if (fabs(particle.x - landmark.x_f) <= sensor_range &&
       fabs(particle.y - landmark.y_f) <= sensor_range)
      {
       // LandmarkObs landmarkObs{landmark.id_i, landmark.x_f, landmark.y_f};
        predicted_landmarks.push_back(LandmarkObs{landmark.id_i, landmark.x_f, landmark.y_f});
      }
    }

    dataAssociation(predicted_landmarks, translated_observations);

    for (auto observation: translated_observations)
    {
      auto landmark = map_landmarks.landmark_list[observation.id-1];
      assert(observation.id == landmark.id_i);

      auto multi = multiv_prob(std_landmark[0], std_landmark[1], observation.x, observation.y, landmark.x_f, landmark.y_f);
      particle.weight *= multi;
    }

    if (particle.weight > 1)
      std::cout << particle.weight << std::endl;

    weights.push_back(particle.weight);
  }

}

void ParticleFilter::resample() {
  /**
   * Resample particles with replacement with probability proportional
   *   to their weight. 
   */

  std::default_random_engine gen;

  std::vector<Particle> resamples(num_particles);

  for (auto i = 0; i < num_particles; ++i) {
    std::discrete_distribution<decltype(i)> index(weights.begin(), weights.end());
    resamples[i] = particles[index(gen)];
  }

  particles = std::move(resamples);

}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}