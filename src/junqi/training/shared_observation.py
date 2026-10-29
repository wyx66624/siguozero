"""Bounded shared observation rows; messages retain independent array ownership."""
import mmap
import os
from pathlib import Path
import tempfile
import numpy as np

class SharedObservationBuffer:
    def __init__(self,descriptor,*,temporary=None):
        self.descriptor=descriptor;self.temporary=temporary
        self.file=open(descriptor[0],'r+b') if temporary is None else None
        stream=self.file if temporary is None else temporary
        shape=tuple(descriptor[1]);size=int(np.prod(shape))*2
        self.mapping=mmap.mmap(stream.fileno(),size,access=mmap.ACCESS_WRITE)
        self.rows=np.ndarray(shape,dtype=np.int16,buffer=self.mapping)

    @classmethod
    def create(cls,pool):
        from .history_arrays import observation_width
        game=pool.slots[0].game
        shape=(2,len(pool.slots),game.config.player_count,observation_width(pool.mode,game.config.dead_rules_enabled)+14)
        directory='/dev/shm' if Path('/dev/shm').is_dir() else None
        temporary=tempfile.NamedTemporaryFile(prefix='junqi_observations_',dir=directory)
        temporary.truncate(int(np.prod(shape))*2);temporary.flush()
        return cls((temporary.name,shape),temporary=temporary)

    def pack(self,rows):
        result=[]
        for row in rows:
            self.rows[0,row.index]=row.records
            central=row.centralized_records is not None
            if central:self.rows[1,row.index]=row.centralized_records
            result.append(row._replace(records=None,centralized_records=True if central else None))
        return result

    def unpack(self,rows):
        return [row._replace(records=self.rows[0,row.index].copy(),
                 centralized_records=self.rows[1,row.index].copy() if row.centralized_records is True else None)
                if row.records is None else row for row in rows]

    def close(self):
        self.rows=None
        self.mapping.close()
        if self.file is not None:self.file.close()
        if self.temporary is not None:self.temporary.close()
