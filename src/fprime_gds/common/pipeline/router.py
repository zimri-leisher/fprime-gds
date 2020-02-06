"""
router.py:

Sets up a router object to handle incoming and outgoing messages adding in handshake token to the front of the message
that can then be sent back as a handshake packet. This router will allow returning handshake callback end up at the
originating object correctly.

@author mstarch
"""
import struct
import fprime_gds.common.handlers


class OutgoingRouter(fprime_gds.common.handlers.DataHandler):
    """
    Handshake router that inspects the originating caller and notes it in a table. This table can then be consulted when
    the data returns for sending handshakes back to the originating object.
    """
    def __init__(self):
        """
        Constructor of the outgoing router.
        """
        self.consumers = []

    def register(self, consumer):
        """
        Register the consumers of the outgoing data.
        """
        self.consumers.append(consumer)

    def data_callback(self, data, sender=None):
        """
        Handles incoming data by stamping on a handshake token to be passed back in the handshake packet. This reads
        from the sender parameter to creat this token, otherwise "0000" is sent out.
        :param data: encoded data to be prepended to
        :param sender: sender id to append to.
        """
        if sender is None:
            sender = 0
        elif isinstance(sender, int):
            raise ValueError("Sender expected to integer")
        bytes_data = data + struct.unpack(">I", sender)
        for consumer in self.consumers:
            consumer.data_callback(bytes_data)


class IncomingRouter(fprime_gds.common.handlers.DataHandler):
    """
    Handshake router that inspects the returning handshake packet, and routes back to the originating object found in
    the outgoing router.
    """
    def __init__(self):
        """
        Constructor of the outgoing router.
        """
        self.consumers = []

    def register(self, consumer):
        """
        Register the consumers of the outgoing data.
        """
        self.consumers.append(consumer)

    def data_callback(self, data, sender=None):
        """
        Handles incoming data by stamping on a handshake token to be passed back in the handshake packet. This reads
        from the sender parameter to creat this token, otherwise "0000" is sent out.
        :param data: encoded data to be prepended to
        :param sender: sender id to append to.
        """
        if sender is None:
            sender = 0
        elif isinstance(sender, int):
            raise ValueError("Sender expected to integer")
        bytes_data = data + struct.unpack(">I", sender)
        for consumer in self.consumers:
            consumer.data_callback(bytes_data)