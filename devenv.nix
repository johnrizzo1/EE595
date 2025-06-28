{ pkgs, lib, config, inputs, ... }:

{
  name = "EE595";

  packages = with pkgs; [ 
    git
  ];

  languages.python.enable = true;
  languages.python.version = "3.12";
  languages.python.poetry.enable = true;
  # languages.python.uv.enable = true;
  languages.python.venv.enable = true;
  # # languages.python.venv.quiet = true;
  # languages.python.venv.requirements = ''
  #   alembic
  #   datasets
  #   databento
  #   huggingface_hub
  #   jupyter
  #   jupyterlab
  #   keras
  #   keras-cv
  #   keras-hub
  #   matplotlib
  #   numpy
  #   pandas
  #   python-dotenv
  #   scikit-learn
  #   seaborn
  #   sqlalchemy
  #   tensorflow
  #   torch
  # '';

  dotenv.enable = true;
  dotenv.disableHint = true;
  delta.enable = true;
  difftastic.enable = true;
  devenv.warnOnNewVersion = false;
  # See full reference at https://devenv.sh/reference/options/
}
