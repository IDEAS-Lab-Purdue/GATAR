import env.grids
import yaml

# read configs from yaml file

configs=yaml.load(open('config/test.yaml','r'),Loader=yaml.FullLoader)

# create environment
instance = env.grids.gridWorld(configs['env'])


