# Author: Mikita Sazanovich

import utils
from dvae.trainer import Trainer


def main():
  args = utils.parse_args()
  config = utils.load_config(args.config_path)
  trainer = Trainer(config, args.name, args.seed)
  trainer.train()


if __name__ == '__main__':
  main()
