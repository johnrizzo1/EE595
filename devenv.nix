{ pkgs, lib, config, inputs, ... }:

{
  packages = with pkgs; [ 
    git
  ];

  languages.python.enable = true;
  languages.python.version = "3.12";
  languages.python.uv.enable = true;
  languages.python.venv.enable = true;
  languages.python.venv.quiet = true;
  languages.python.venv.requirements = ''
    alembic
    datasets
    huggingface_hub
    jupyter
    jupyterlab
    keras
    keras-cv
    keras-hub
    matplotlib
    numpy
    pandas
    python-dotenv
    seaborn
    scikit-learn
    sqlalchemy
    tensorflow
    torch
  '';

  dotenv.enable = true;
  dotenv.disableHint = true;
  delta.enable = true;
  difftastic.enable = true;
  devenv.warnOnNewVersion = false;
  # See full reference at https://devenv.sh/reference/options/
}
