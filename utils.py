import termcolor


def success(str):
  return termcolor.colored(str, 'green')


def warning(str):
  return termcolor.colored(str, 'yellow')


def danger(str):
  return termcolor.colored(str, 'red')


def log_args(args):
  print(warning('arguments:'))
  for key, value in sorted(vars(args).items(), key=lambda kv: kv[0]):
    print(warning('\t{}:').format(key), value)
