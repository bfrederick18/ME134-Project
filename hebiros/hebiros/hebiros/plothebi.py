'''plothebi.py

   Plot the /joint_states and /joint_commands recorded in the ROS2 bag.
'''

import rclpy
import numpy as np
import matplotlib.pyplot as plt

import glob, os, sys

from rosbag2_py                 import SequentialReader
from rosbag2_py._storage        import StorageOptions, ConverterOptions
from rclpy.serialization        import deserialize_message

from sensor_msgs.msg            import JointState


#
#  Extract Joint Data
#
def jointdata(msgs, t0, jointname):
    # Make sure we have data
    if not msgs:
        raise ValueError("No data!")

    # Grab the names (assuming all will be the same).
    names = msgs[0].name

    # Grab the time.
    sec  = np.array([msg.header.stamp.sec     for msg in msgs])
    nano = np.array([msg.header.stamp.nanosec for msg in msgs])
    t    = sec + nano*1e-9 - t0

    # Grad the data.
    pos = np.array([msg.position for msg in msgs])
    vel = np.array([msg.velocity for msg in msgs])
    eff = np.array([msg.effort   for msg in msgs])

    # Make sure we have data.
    (M,N) = (len(msgs), len(names))
    if np.shape(pos)[1] == 0:  pos = np.full((M,N), np.nan)
    if np.shape(vel)[1] == 0:  vel = np.full((M,N), np.nan)
    if np.shape(eff)[1] == 0:  eff = np.full((M,N), np.nan)

    # Extract the specified joint.
    if jointname != 'all':
        # Grab the joint index.
        try:
            index = int(jointname)
        except Exception:
            index = None
        try:
            if index:
                jointname = names[index]
            else:
                index = names.index(jointname)
        except Exception:
            raise ValueError("No data for joint '%s'" % jointname)

        # Limit the data.
        names = [names[index]]
        pos   = pos[:,index]
        vel   = vel[:,index]
        eff   = eff[:,index]

    # Return the data.    
    return (names, t, pos, vel, eff)


#
#  Plot the Joint Actual and Command Data
#
def plotboth(actmsgs, cmdmsgs, t0, title, jointname='all'):
    # Process the actual and command messages.
    (nact, tact, pact, vact, eact) = jointdata(actmsgs, t0, jointname)
    (ncmd, tcmd, pcmd, vcmd, ecmd) = jointdata(cmdmsgs, t0, jointname)

    # Make sure the names match!
    if nact != ncmd:
        raise ValueError("Joint names in actual/command data do not match!")

    # Re-zero the start time.
    tstart = min(min(tact), min(tcmd))
    print("Starting at time ", tstart)
    tact = tact - tstart
    tcmd = tcmd - tstart

    # Create a figure to plot pos/vel/eff vs. t
    fig, axs = plt.subplots(3, 1)

    # Plot the data in the subplots.
    axs[0].plot(tact, pact, linestyle='-' )
    axs[0].set_prop_cycle(None)
    axs[0].plot(tcmd, pcmd, linestyle='--')
    axs[0].set(ylabel='Position (rad)')

    axs[1].plot(tact, vact, linestyle='-' )
    axs[1].set_prop_cycle(None)
    axs[1].plot(tcmd, vcmd, linestyle='--')
    axs[1].set(ylabel='Velocity (rad/sec)')

    axs[2].plot(tact, eact, linestyle='-' )
    axs[2].set_prop_cycle(None)
    axs[2].plot(tcmd, ecmd, linestyle='--')
    axs[2].set(ylabel='Effort (Nm)')

    # Connect the time.
    axs[1].sharex(axs[0])
    axs[2].sharex(axs[0])
    axs[2].set(xlabel='Time (sec)')

    # Add the title and legend.
    fig.suptitle(title)
    axs[0].legend(nact, ncol=len(nact),
                  loc='lower center', bbox_to_anchor=(0.5, 1.0))

    # Draw grid lines and allow only "outside" ticks/labels in each subplot.
    for ax in axs.flat:
        ax.grid()
        ax.label_outer()


#
#  Plot the Joint Data
#
def plotsolo(jntmsgs, t0, title, jointname='all'):
    # Process the joint messages.
    (n, t, p, v, e) = jointdata(jntmsgs, t0, jointname)

    # Re-zero the start time.
    tstart = min(t)
    print("Starting at time ", tstart)
    t = t - tstart

    # Create a figure to plot pos/vel/eff vs. t
    fig, axs = plt.subplots(3, 1)

    # Plot the data in the subplots.
    axs[0].plot(t, p)
    axs[0].set(ylabel='Position (rad)')

    axs[1].plot(t, v)
    axs[1].set(ylabel='Velocity (rad/sec)')

    axs[2].plot(t, e)
    axs[2].set(ylabel='Effort (Nm)')

    # Connect the time.
    axs[1].sharex(axs[0])
    axs[2].sharex(axs[0])
    axs[2].set(xlabel='Time (sec)')

    # Add the title and legend.
    fig.suptitle(title)
    axs[0].legend(n, ncol=len(n),
                  loc='lower center', bbox_to_anchor=(0.5, 1.0))

    # Draw grid lines and allow only "outside" ticks/labels in each subplot.
    for ax in axs.flat:
        ax.grid()
        ax.label_outer()


#
#  Main Code
#
def main():
    # Grab the arguments.
    jointname = 'all'    if len(sys.argv) < 3 else sys.argv[2]
    bagname   = 'latest' if len(sys.argv) < 2 else sys.argv[1]

    # Check for the latest ROS bag:
    if bagname == 'latest':
        # Report.
        print("Looking for latest ROS bag...")

        # Look at all bags, making sure we have at least one!
        dbfiles = glob.glob('*/*.db3')
        if not dbfiles:
            raise FileNoFoundError('Unable to find a ROS2 bag')

        # Grab the modification times and the index of the newest.
        dbtimes = [os.path.getmtime(dbfile) for dbfile in dbfiles]
        i = dbtimes.index(max(dbtimes))

        # Select the newest.
        bagname = os.path.dirname(dbfiles[i])

    # Report.
    print("Reading ROS bag '%s'"  % bagname)
    print("Processing joint '%s'" % jointname)


    # Set up the BAG reader.
    reader = SequentialReader()
    try:
        reader.open(StorageOptions(uri=bagname, storage_id='sqlite3'),
                    ConverterOptions('', ''))
    except Exception as e:
        print("Unable to read the ROS bag '%s'!" % bagname)
        print("Does it exist and WAS THE RECORDING Ctrl-c KILLED?")
        raise OSError("Error reading bag - did recording end?") from None

    # Get the starting time.
    t0 = reader.get_metadata().starting_time.nanoseconds * 1e-9 - 0.01

    # Get the topics and types:
    print("The bag contains messages for:")
    for x in reader.get_all_topics_and_types():
        print("  topic %-20s of type %s" % (x.name, x.type))


    # Pull out the relevant messages.
    actmsgs = []
    cmdmsgs = []
    while reader.has_next():
        # Grab a message.
        (topic, rawdata, timestamp) = reader.read_next()

        # Pull out the deserialized message.
        if   topic == '/joint_states':
            actmsgs.append(deserialize_message(rawdata, JointState))
        elif topic == '/joint_commands':
            cmdmsgs.append(deserialize_message(rawdata, JointState))


    # Process the actual/command joint data
    if actmsgs and cmdmsgs:
        print("Plotting actual/command data...")
        title = "Actual/Command Data in '%s'" % bagname
        plotboth(actmsgs, cmdmsgs, t0, title, jointname)

    elif actmsgs:
        print("Plotting actual data...")
        title = "Actual Data in '%s'" % bagname
        plotsolo(actmsgs, t0, title, jointname)
    elif cmdmsgs:
        print("Plotting command data...")
        title = "Command Data in '%s'" % bagname
        plotsolo(cmdmsgs, t0, title, jointname)
        
    # Show
    plt.show()


#
#   Run the main code.
#
if __name__ == "__main__":
    main()
