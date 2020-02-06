"""
@brief Encoder for file data

This encoder takes in file objects, serializes them, and sends the results
to all registered senders.

Serialized command format:
    +--------------------------------+          -
    | Header = "A5A5 "               |          |
    | (5 byte string)                |          |
    +--------------------------------+      Added by
    | Destination = "GUI " or "FSW " |       Sender
    | (4 byte string)                |          |
    +--------------------------------+          -
    | Command descriptor             |
    | (0x????????)                   |
    | (4 byte number)                |
    +--------------------------------+
    | Length of descriptor, opcode,  |
    | and argument data              |
    | (4 bytes)                      |
    +--------------------------------+
    | Packet type                    |
    | (4 bytes)                      |
    +--------------------------------+
    | Sequence ID                    |
    | (4 bytes)                      |
    +--------------------------------+
    | Type specific                  |
    +--------------------------------+
    | Type specific                  |
    +--------------------------------+
    | ...                            |
    +--------------------------------+
    | Type specific                  |
    +--------------------------------+

@date Created February 2020-02-05
@author mstarch

@bug No known bugs
"""
from __future__ import absolute_import

import struct

from . import encoder
from fprime_gds.common.data_types.file_data import FilePacketType


class FileEncoder(encoder.Encoder):
    """
    Encodes the file data. This plugs into the uplink system and allows for data to be sent to the spacecraft.
    """
    def __init__(self, dest="FSW", config=None):
        """
        Constructs a file encoder. Defaults to FSW as destination..

        Args:
            dest (string, "FSW" or "GUI", default="FSW"): Destination for binary
                  data produced by encoder.
            config (ConfigManager, default=None): Object with configuration data
                    for the sizes of fields in the binary data. If None passed,
                    defaults are used.
        """
        super(FileEncoder, self).__init__(dest, config)

        self.len_obj = self.config.get_type("msg_len")

    def encode_api(self, data):
        """
        Encodes specific file packets. This will allow the data to be sent out.
        :param data: FilePacket type to send.
        :return: encoded bytes data
        """
        out_data = struct.pack(">BI", data.packetType, data.seqID)
        # Packet Type determines the variables following the seqID
        if data.packetType == FilePacketType.START:
            out_data += struct.pack(">IIBsBs", 2 + len(data.sourcePath) + len(data.destPath),
                                    len(data.sourcePath), data.sourcePath, len(data.destPath), data.destPath)
        elif data.packetType == FilePacketType.DATA:
            out_data += struct.pack(">IH", data.offset, data.length)
            out_data += data.dataVar
        elif data.packetType == FilePacketType.END:
            out_data += struct.pack(">I", data.hashValue)
        elif data.packetType != FilePacketType.CANCEL:
            raise Exception("Invalid packet type found while encoding: {}".format(data.packetType))
        return out_data
