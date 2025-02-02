#!/usr/bin/env python3
#
#   hebiscan.py
#
#   Scan for HEBI actuators and report.
#
#   Note this does not use any ROS functionality.
#
import hebi

from time import sleep


#
#   HEBI Class
#
#   Simply create a Lookup object to determine/locate the connected
#   actuators.  Report as requested.
#
class Hebi():
    # Initialization.
    def __init__(self):
        # Locate HEBI actuators on the network, waiting 1s for discovery.
        self.lookup = hebi.Lookup()
        sleep(1)

    # Reporting
    def reportActuators(self):
        # Print the results. 
        print('HEBI actuators found on the local network:')
        for entry in self.lookup.entrylist:
            # Extract the family/name/address
            family  = entry.family
            name    = entry.name
            address = entry.mac_address

            # Print...
            print("family %s name %s address %s" % (family, name, address))


#
#   Main Code
#
def main(args=None):
    # Instantiate the HEBI object.
    hebi = Hebi()

    # Report the actuators.
    hebi.reportActuators()

if __name__ == "__main__":
    main()
