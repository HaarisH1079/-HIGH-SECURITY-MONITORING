import numpy as np
from collections import OrderedDict

class RoomDetector:
    def __init__(self, db_manager=None, max_disappeared=20, zone_names=None):
        self.nextObjectID = 0
        self.objects = OrderedDict() # ID -> (cX, cY)
        self.disappeared = OrderedDict() # ID -> frames_disappeared
        self.object_zones = OrderedDict() # ID -> current zone
        self.maxDisappeared = max_disappeared
        
        # Default zone names if none provided
        self.zone_names = zone_names if zone_names else ["Room 1", "Hallway", "Room 2"]

    def get_zone(self, x_pos, frame_width):
        """Determine which zone a position is in"""
        margin = 0.20 * frame_width
        if x_pos < margin:
            return self.zone_names[0]
        elif x_pos > (frame_width - margin):
            return self.zone_names[2]
        else:
            return self.zone_names[1]

    def register(self, centroid, frame_width):
        oid = self.nextObjectID
        self.objects[oid] = centroid
        self.disappeared[oid] = 0
        self.object_zones[oid] = self.get_zone(centroid[0], frame_width)
        self.nextObjectID += 1
        return oid

    def deregister(self, objectID):
        last_pos = self.objects[objectID]
        last_zone = self.object_zones.get(objectID, "Hallway")
        del self.objects[objectID]
        del self.disappeared[objectID]
        if objectID in self.object_zones:
            del self.object_zones[objectID]
        return last_pos, last_zone

    def update(self, rects, frame_width):
        """
        rects: list of (x, y, w, h)
        frame_width: int
        Returns: list of events [(event_type, objectID, location, data)]
        event_type: "ZONE_CHANGE", "ENTERED_ROOM", "EXITED_ROOM"
        """
        inputCentroids = np.zeros((len(rects), 2), dtype="int")
        for (i, (x, y, w, h)) in enumerate(rects):
            cX = int(x + w / 2.0)
            cY = int(y + h / 2.0)
            inputCentroids[i] = (cX, cY)

        events = []
        margin = 0.20 * frame_width

        # If no objects currently tracked
        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                oid = self.register(inputCentroids[i], frame_width)
                zone = self.object_zones[oid]
                # Generate entry event for the zone they appeared in
                if zone in ["Room 1", "Room 2"]:
                    events.append(("EXITED_ROOM", oid, zone, i))
                else:
                    events.append(("ZONE_CHANGE", oid, zone, i))
            return events

        # If no input rects but we have objects
        if len(rects) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    last_pos, last_zone = self.deregister(objectID)
                    # They disappeared - log where they were last seen
                    if last_zone in ["Room 1", "Room 2"]:
                        events.append(("ENTERED_ROOM", objectID, last_zone, None))
            return events

        # Standard Tracking
        objectIDs = list(self.objects.keys())
        objectCentroids = list(self.objects.values())

        D = np.zeros((len(objectCentroids), len(inputCentroids)))
        for i in range(len(objectCentroids)):
            for j in range(len(inputCentroids)):
                D[i, j] = np.linalg.norm(np.array(objectCentroids[i]) - inputCentroids[j])

        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        usedRows = set()
        usedCols = set()

        for (row, col) in zip(rows, cols):
            if row in usedRows or col in usedCols:
                continue

            if D[row, col] > frame_width * 0.5:
                continue

            objectID = objectIDs[row]
            old_zone = self.object_zones.get(objectID, "Hallway")
            new_zone = self.get_zone(inputCentroids[col][0], frame_width)
            
            self.objects[objectID] = inputCentroids[col]
            self.disappeared[objectID] = 0
            
            # Check for zone change
            if old_zone != new_zone:
                self.object_zones[objectID] = new_zone
                events.append(("ZONE_CHANGE", objectID, new_zone, col))

            usedRows.add(row)
            usedCols.add(col)

        # Handle disappeared objects
        unusedRows = set(range(0, D.shape[0])).difference(usedRows)
        for row in unusedRows:
            objectID = objectIDs[row]
            self.disappeared[objectID] += 1
            if self.disappeared[objectID] > self.maxDisappeared:
                last_pos, last_zone = self.deregister(objectID)
                if last_zone in ["Room 1", "Room 2"]:
                    events.append(("ENTERED_ROOM", objectID, last_zone, None))

        # Register new objects
        unusedCols = set(range(0, D.shape[1])).difference(usedCols)
        for col in unusedCols:
            oid = self.register(inputCentroids[col], frame_width)
            zone = self.object_zones[oid]
            if zone in ["Room 1", "Room 2"]:
                events.append(("EXITED_ROOM", oid, zone, col))
            else:
                events.append(("ZONE_CHANGE", oid, zone, col))

        return events

    def get_objects(self):
        return self.objects
    
    def get_object_zone(self, objectID):
        return self.object_zones.get(objectID, "Hallway")
