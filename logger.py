import tensorflow as tf

class Logger(object):
    """TensorBoard logger using TF2 summary API."""

    def __init__(self, log_dir):
        """Initialize the summary writer."""
        # Create a TF2 summary writer
        self.writer = tf.summary.create_file_writer(log_dir)

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        # Use the writer as a context manager to record summaries
        with self.writer.as_default():
            tf.summary.scalar(name=tag, data=value, step=step)
            self.writer.flush()