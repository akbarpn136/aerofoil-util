import typer


def main(name: str, email: str, formal: bool = False):
    if formal:
        typer.echo(f"Salam kenal, {name} - {email}")
    else:
        typer.echo(f"Whats up.., {name} - {email}")


if __name__ == "__main__":
    typer.run(main)
