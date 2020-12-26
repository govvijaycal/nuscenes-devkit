# NuScenes Python SDK Notes

---
## Prediction

### PredictHelper
  * Extracts prediction information given an sample/instance token
  * Gets the past/future for an agent or sample (i.e. feature/label for prediction)
  * Gets pose, velocity (norm diff xy), acceleration, yaw rate for an agent
  * Converts between local and global coordinate systems
  * .get_sample_annotation -> maps sample/instance (one agent at a given timestamp) to a record
  * .get_annotations_for_sample -> for a given sample, gets a list of records for all relevant agents in that timestamp
  * .get_past_or_future_for_sample -> returns a dictionary of "time series" records for each agent in a given sample
  
### Input Representation

Refer to map_expansion/map_api.py for how the NuScenesMap class works.  Essentially, these are the key layers:

`self.non_geometric_polygon_layers = ['drivable_area', 'road_segment', 'road_block', 'lane', 'ped_crossing',
                                             'walkway', 'stop_line', 'carpark_area']`

`self.non_geometric_line_layers = ['road_divider', 'lane_divider', 'traffic_light']`

#### interface.py
  * Abstract base class implementations for StaticLayerRepresentation, AgentRepresentation, Combinator
  * Class implementaiton for InputRepresentation which runs .make_representation and then .combine methods of above classes.

#### combinators.py
  * Provides a Rasterizer class implementation (inherits from Combinator).
  * Rasterizer simply takes a set of 3-channel images and makes a single combined 3-channel image (using reduce on a add_foreground_to_image helper function).
  * The idea is to each subsequent image in the list as a "foreground" image that has precedence but maintaining the "background" images wherever the "foreground" image is not applicable.

#### agents.py
  * Main class is AgentBoxesWithFadedHistory, inheriting from AgentRepresentation.  Specify the history frequency and duration, and same resolution/extent parameters as with static layers.  A color map is used to assign objects varying colors.
  * Makes the ego agent red and other vehicles yellow, objects violet, human/animals orange.  Draws oriented bboxes (see relevant helper functions) with cv.fillPoly.
  * Faded history adjusts the HSV of the base color so that the value increases as you get to the latest time step, it's darker earlier in the history.  See fade_color function for details.
  * These links are helpful to understand the label categories:
    * https://github.com/nutonomy/nuscenes-devkit/blob/master/docs/instructions_nuscenes.md#labels
    * https://www.nuscenes.org/nuscenes#data-annotation
  * From my understanding, traffic lights are only annotated for location/bulb location.  The dynamic state of the traffic light is not maintained.

#### static_layers.py
  * Handles all aspect of the static map layers.  Uses the map_expansion utils to get relevant features.  First draws ego in the center in a larger image and then crops so it follows specified distances.
  * You need to specify **layer_names** and **colors** to choose which layers you want to see, along with their RGB values.
  * Default is "driveable_area" (white), "ped_crossing" (gray-blue), "walkway" (blue).  See the map_expansion for details.
  * Default settings are 10 cm / pixel, 40 m ahead and 10 m behind, +/- 25 m to the sides.
  * Key map function is .get_map_mask which takes in an oriented bbox (aka patch), layer names, canvas_size and returns a multi-channel CxHxW image.
  * Also handles the lanes via draw_lanes_in_agent_frame function.  It draws all lanes within a specified radius and orients according to ego's pose.  This is done simply by connecting line segments at a resolution of 1 m.  The color is given by the function color_by_yaw which simply uses the relative pose (with ego aligned with the y-axis) to pick a hue value (blue = 0, purple = 90, red = 180, green = 270).
  * The representation output is an oriented and cropped RGB image with all static layers combined in a priority order.

#### utils.py
  * Helper functions to extract oriented crops and do pixel/global coord transforms

### Models

#### physics.py
  * Extract full kinematic state [x, y, theta, xd, yd, xdd, ydd, v, wz, acc]
  * Bank of Models:
    * Constant Velocity and Heading
    * Constant Acceleration and Heading
    * Constant Speed and Yaw Rate
    * Constant Acceleration and Yaw Rate
  * Physics Oracle takes the bank of model and outputs multiple trajectory hypotheses.  For evaluation, the closest (L2 norm) trajectory is given as the prediction.


#### backbone.py
 * Implements MobileNetv2 and ResNet backbones.
 * Cuts off the final softmax layer by default.

#### covernet.py
 * CoverNet class
   * Also a simple CNN architecture with a bunch of linear layers (not FC?).
   * Treated as a classification problem, so just output logits.
 * ConstantLatticeLoss
   * lattice = set of trajectories wrt which we are doing classification.
   * Basically just a cross_entropy loss where the label is the nearest (L2 norm) lattice trajectory.
   * Overall quite simple to use once you have a lattice + similarity metric defined.

#### mtp.py
 * MTP class is a simple CNN: backbone + 2 additional FC layers.
 * Trajectory prediction = 2 Hz, 6 seconds, 2 states (XY) -> 24.
 * Output is a single vector containing num_modes * (pred_size + 1) elements.  So it's a concatenation of the trajectory prediction and associated probabilities.
 * MTPLoss is the interesting class.  
   * Handles association of the ground truth trajectory to a predicted mode by looking at the angle between last states of the trajectory (cosine similarity) as a first threshold and then using L2 norm to finally give a best mode.
   * Regression loss = smooth l1 loss wrt best mode trajectory
   * Classification loss = cross entropy with one-hot encoding of best mode
   * From MultiPath paper, one challenge is mode collapse when training such an architecture (sort of like an iterative optimization where convergence is not guaranteed).

---
