import argparse
import os



def argparser():
  parser = argparse.ArgumentParser()
  # for model
  parser.add_argument(
      '--smi_window_lengths',
      type=int,
      nargs='+',
      help='Space seperated list of motif filter lengths. (ex, --window_lengths  8 16)'
  )
  parser.add_argument(
      '--num_windows',
      type=int,
      nargs='+',
      help='Space seperated list of the number of motif filters corresponding to length list. (ex, --num_windows 32 128)'
  )
  parser.add_argument(
      '--num_hidden',
      type=int,
      default=0,
      help='Number of neurons in hidden layer.'
  )
   
  parser.add_argument(
      '--max_smi_len',
      type=int,
      default=0,
      help='Length of drug sequences.'
  )
  # for learning
  parser.add_argument(
      '--learning_rate',
      type=float,
      default=0.001,
      help='Initial learning rate.'
  )
  parser.add_argument(
      '--num_epoch',
      type=int,
      default=500,
      help='Number of epochs to train.(250,500,1000)'
  )
  parser.add_argument(
      '--batch_size',
      type=int,
      default=706,
      help='Batch size. Must divide evenly into the dataset sizes.'
  )
  parser.add_argument(
      '--dataset_path',
      type=str,
      default='/data/kiba/',
      help='Directory for input data.'
  )
  parser.add_argument(
      '--problem_type',
      type=int,
      default=1,
      help='Type of the prediction problem '
  )

  parser.add_argument(
      '--is_log',
      type=int,
      default=0,
      help='use log transformation for Y'
  )
  parser.add_argument(
      '--checkpoint_path',
      type=str,
      default='',
      help='Path to write checkpoint file.'
  )
  parser.add_argument(
      '--log_dir',
      type=str,
      default='/tmp',
      help='Directory for log data.'
  )
 
  FLAGS, unparsed = parser.parse_known_args()

  
  return FLAGS


def logging(msg, FLAGS):
  fpath = os.path.join( FLAGS.log_dir, "log.txt" )
  with open( fpath, "a" ) as fw:
    fw.write("%s\n" % msg)
 
