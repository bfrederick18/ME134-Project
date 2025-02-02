function  plottaskdata(bagfoldername)
%
%   plottaskdata(bagfoldername)
%
%   Plot the /pose and /twist topics saved in the bag folder.  If
%   'bagfoldername' is not given or given as 'latest', use the most
%   recent bag folder.
%
%   Also return the time/pos/vel/quaternion/omega data, with each row
%   representing one sample-time, the columns being x/y/z.
%

%
%   Check the arguments
%
% If no bagfile is specified, use the most recent.
if (~exist('bagfoldername') || strcmp(bagfoldername, 'latest'))
    bagfoldername = latestbagfoldername();
end

%
%   Read the data.
%
% Load the bag.
try
    % Note: renamed to ros2bagreader() in R2022b
    bag = ros2bag(bagfoldername);
catch ME
    % Check whether the bag is incomplete.
    if (strcmp(ME.identifier, 'ros:mlros2:bag:YAMLFileNotFound'))
        disp('Recording not complete...  Is the recording stopped?');
        rethrow(ME);
        
    % Otherwise, rethrow the error.
    else
        rethrow(ME);
    end
end

% Grab the bag's start time in seconds.  Go back 10ms, as the first
% message may originated one cycle earlier.
tbag = double(bag.StartTime);
if tbag > 1e14
    tbag = tbag * 1e-9;         % If nanoseconds (R2022), convert
end
t0 = tbag - 0.010;

% Pull the data from the messages in the bag.
[tp, p, q] = posedata( bagmsgs(bag, '/pose'));
[tv, v, w] = twistdata(bagmsgs(bag, '/twist'));

% Shift the initial time, to be relative to the bag's start time.
tp = tp - t0;
tv = tv - t0;


%
%   Plot Translation.
%
% Prepare the figure.
figure(2);
clf;

% Plot.
ax(1) = subplot(2,1,1);
plot(tp, p, 'LineWidth', 2);
grid on;
ylabel('Position (m)');
title(sprintf('Task Translation in ''%s''', bagfoldername), ...
      'Interpreter', 'none');
legend({'x', 'y', 'z'});

ax(2) = subplot(2,1,2);
plot(tv, v, 'Linewidth', 2);
grid on;
ylabel('Velocity (m/s)');
xlabel('Time (sec)');

linkaxes(ax, 'x');

% Name the figure and span the full 8.5x11 page.
set(gcf, 'Name',          'Task Translation Data');
set(gcf, 'PaperPosition', [0.25 0.25 8.00 10.50]);

% Return to the top subplot, so subsequent title()'s go here...
subplot(2,1,1);


%
%   Plot Orientation.
%
% Prepare the figure.
figure(3);
clf;

% Plot.
ax(1) = subplot(2,1,1);
plot(tp, q, 'LineWidth', 2);
grid on;
ylabel('Quaternion');
title(sprintf('Task Orientation in ''%s''', bagfoldername), ...
      'Interpreter', 'none');
legend({'x', 'y', 'z', 'w'});

ax(2) = subplot(2,1,2);
plot(tv, w, 'Linewidth', 2);
grid on;
ylabel('Angular Velocity (rad/sec)');
xlabel('Time (sec)');
legend({'x', 'y', 'z'});

linkaxes(ax, 'x');

% Name the figure and span the full 8.5x11 page.
set(gcf, 'Name',          'Task Orientation Data');
set(gcf, 'PaperPosition', [0.25 0.25 8.00 10.50]);

% Return to the top subplot, so subsequent title()'s go here...
subplot(2,1,1);

end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function bagfoldername = latestbagfoldername()
%
%   bagfoldername = latestbagfoldername()
%
%   Return the name of the latest bag folder including a bag file.
%   Error if there are no bag folders.
%

% Get a list of bag files in subfolders of the current folder.
d = dir('*/*.db3');

% Make sure we have at least one bag file.
if (~size(d,1))
    error('Unable to find a bag folder (including a bag file)');
end

% Find the most recently modified bag file.
[~, idx] = max([d.datenum]);

% Determine the folder that holds the bag file.
[root, name, ext] = fileparts(d(idx).folder);
bagfoldername = strcat(name,ext);

% Report.
disp(['Using bag folder ''' bagfoldername '''']);

end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function  msgs = ros2bagmsgs(bagfoldername, topicname)
%
%   msgs = ros2bagmsgs(bagfoldername, topicname)
%
%   Extract the messages of the named topic from the bag file in the
%   give folder.  The messages are returned as a struct array.  The
%   structure contains MessageType as well as the fields of the topic.
%

% Load the bag.
try
    % Note: renamed to ros2bagreader() in R2022b
    bag = ros2bag(bagfoldername);
catch
    error(['Unable to open the bag folder ''' bagfoldername '''']);
end

% Grab the messages.
msgs = bagmsgs(bag, topicname);

end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function  msgs = bagmsgs(bag, topicname)
%
%   msgs = bagmsgs(bag, topicname)
%
%   Extract the messages of the named topic from the given ROS2 bag.
%   The messages are returned as a struct array.  The structure
%   contains MessageType as well as the fields of the topic.
%

% Isolate the specified topic.
topic = select(bag, 'Topic', topicname);
if (~topic.NumMessages)
    warning(['No messages under topic ''' topicname '''']);
end

% Convert the messages in the topic into structure array.
msgs = cell2mat(readMessages(topic));

end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function  [tp, p, q] = posedata(posemsgs)
%
%   [tp, p, q] = posedata(posemsgs)
%
%   Extract the data from the given PoseStamped messages.  Time is
%   absolute.  The return data gives a row per time sample, and a
%   column per axis (x,y,z - and w for quaternion).
%

% Double-check the type.
if (~strcmp(posemsgs(1).MessageType, 'geometry_msgs/PoseStamped'))
    error(['Pose msgs are not of type geometry_msgs/PoseStamped']);
end

% Check the number of samples.
M = length(posemsgs);

% Extract the absolute time (from sec/nsec), do not subtract the first time.
headers = vertcat(posemsgs.header);
stamps  = vertcat(headers.stamp);

sec  = double(vertcat(stamps.sec));
nsec = double(vertcat(stamps.nanosec));
tp   = sec + 1e-9*nsec;

% Extract the data.
poses        = vertcat(posemsgs.pose);
positions    = vertcat(poses.position);
orientations = vertcat(poses.orientation);

p = [vertcat(positions.x) ...
     vertcat(positions.y) ...
     vertcat(positions.z)];

q = [vertcat(orientations.x) ...
     vertcat(orientations.y) ...
     vertcat(orientations.z) ...
     vertcat(orientations.w)];

% Unwrap the quaternion equivalent solutions.
for i = 2:size(q,1)
    if norm(q(i,:) - q(i-1,:)) > 1.5
        q(i,:) = -q(i,:);
    end
end

end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function  [tv, v, w] = twistdata(twistmsgs)
%
%   [tv, v, w] = twistdata(twistmsgs)
%
%   Extract the data from the given TwistStamped messages.  Time is
%   absolute.  The return data gives a row per time sample, and a
%   column per axis (x,y,z).
%

% Double-check the type.
if (~strcmp(twistmsgs(1).MessageType, 'geometry_msgs/TwistStamped'))
    error(['Twist msgs are not of type geometry_msgs/TwistStamped']);
end

% Check the number of samples.
M = length(twistmsgs);

% Extract the absolute time (from sec/nsec), do not subtract the first time.
headers = vertcat(twistmsgs.header);
stamps  = vertcat(headers.stamp);

sec  = double(vertcat(stamps.sec));
nsec = double(vertcat(stamps.nanosec));
tv   = sec + 1e-9*nsec;

% Extract the data.
twists   = vertcat(twistmsgs.twist);
linears  = vertcat(twists.linear);
angulars = vertcat(twists.angular);

v = [vertcat(linears.x) ...
     vertcat(linears.y) ...
     vertcat(linears.z)];

w = [vertcat(angulars.x) ...
     vertcat(angulars.y) ...
     vertcat(angulars.z)];

end
