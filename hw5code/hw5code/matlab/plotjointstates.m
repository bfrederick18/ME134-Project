function  [t, pos, vel] = plotjointstates(bagfoldername, joint)
%
%   [t, pos, vel] = plotjointstates(bagfoldername, joint)
%
%   Plot the /joint_states topic saved in the bag folder.  If
%   'bagfoldername' is not given or given as 'latest', use the most
%   recent bag folder.  If 'joint' is given and not 'all', plot only the
%   joint given by an index (number) or name (string).
%
%   Also return the time/pos/vel data, with each row representing one
%   sample-time, each column being a joint.
%

%
%   Check the arguments
%
% If no bagfile is specified, use the most recent.
if (~exist('bagfoldername') || strcmp(bagfoldername, 'latest'))
    bagfoldername = latestbagfoldername();
end

% If no joint is given, use all joints.
if (~exist('joint'))
    joint = 'all';
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

% Grab the /joint_states messages.
msgs = bagmsgs(bag, '/joint_states');

% Pull the data from the messages.
[t, pos, vel, eff, names] = jointstatedata(msgs, joint);

% Shift the initial time, to be relative to the bag's start time.
t = t - t0;

        
%
%   Plot.
%
% Skip if outputing data.
if (nargout)
    disp('Skipping the plot when outputing data.');
    return;
end

% Prepare the figure.
figure(1);
clf;

% Plot.
ax(1) = subplot(2,1,1);
plot(t, pos, 'LineWidth', 2);
grid on;
ylabel('Position (rad)');
title(sprintf('Joint Data in ''%s''', bagfoldername), ...
      'Interpreter', 'none');
legend(names);

ax(2) = subplot(2,1,2);
plot(t, vel, 'Linewidth', 2);
grid on;
ylabel('Velocity (rad/sec)');
xlabel('Time (sec)');

linkaxes(ax, 'x');

% Name the Figure and span the full 8.5x11 page.
set(gcf, 'Name',          'Joint Data');
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
function  [t, pos, vel, eff, names] = jointstatedata(msgs, joint)
%
%   [t, pos, vel, eff, names] = jointstatedata(msgs, joint)
%
%   Extract the data from the given JointStates messages.  Time is
%   absolute.  The return data gives a row per time sample, and a
%   column per joint (assuming data is available).
%
%   If 'joint' is given and not 'all', return only the joint given by
%   an index (number) or name (string).
%

% Double-check the type.
if (~strcmp(msgs(1).MessageType, 'sensor_msgs/JointState'))
    error(['Messages are not of type sensor_msgs/JointState']);
end

% Check the number of samples and joints (by data).
M    = length(msgs);
Nnam = length(msgs(1).name);
Npos = length(msgs(1).position);
Nvel = length(msgs(1).velocity);
Neff = length(msgs(1).effort);
Nmax = max([Nnam, Npos, Nvel, Neff]);

% Make sure we have named joints.
if (Nnam == 0)
    error(['Messages contain no named joints']);
end

% Extract the names of the joints.
names = msgs(1).name;

% Extract the absolute time (from sec/nsec), do not subtract the first time.
headers = vertcat(msgs.header);
stamps  = vertcat(headers.stamp);

sec  = double(vertcat(stamps.sec));
nsec = double(vertcat(stamps.nanosec));
t    = sec + 1e-9*nsec;

% Extract the msgs, whether the individual elements are row or column vectors.
pos = reshape(horzcat(msgs.position), Npos, M)';
vel = reshape(horzcat(msgs.velocity), Nvel, M)';
eff = reshape(horzcat(msgs.effort)  , Neff, M)';

% Potentially isolate a single joint.
if (exist('joint') && (~strcmp(joint, 'all')))
    % For a numeric joint specification.
    if (isnumeric(joint))
        if (numel(joint) ~= 1)
            error('Bad joint index specification');
        end

        % Grab the joint number and use as index.
        ind = floor(joint);
        if ((ind < 1) || (ind > Nmax))
            error(['Out of range joint index ' num2str(ind)]);
        end
        disp(['Using only joint index ' num2str(ind)]);

    % For a joint name specification.
    elseif (ischar(joint))
        % Find the index for the joint name.
        ind = find(strcmp(names, joint));
        if (isempty(ind))
            error(['Unable to isolate joint ''' joint ''''])
        end
        disp(['Using only joint ''' joint '''']);

    % Otherwise can't do anything.
    else
        error('Bad joint argument');
    end

    % Isolate the data.
    if (ind <= Nnam) names = names(ind); else names = {}; end
    if (ind <= Npos) pos   = pos(:,ind); else pos   = []; end
    if (ind <= Nvel) vel   = vel(:,ind); else vel   = []; end
    if (ind <= Neff) eff   = eff(:,ind); else eff   = []; end
end

end
