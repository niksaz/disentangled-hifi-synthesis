# Author: Mikita Sazanovich

import os

import utils
from idgan.trainer import Trainer


def main():
  args = utils.parse_args()
  config = utils.load_config(args.config_path)
  output_dir = utils.prepare_output_dir(os.path.join('output', args.tag), config)
  utils.fix_random_seed(args.seed)

  trainer = Trainer(config, output_dir)
  trainer.train()


if __name__ == '__main__':
  main()
