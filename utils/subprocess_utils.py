from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
import subprocess, time, os

def run_subprocess(cmd, label="Subprocess Running"):
    with Progress(
        SpinnerColumn(),
        TextColumn(f"[bold]{label}[/bold]"),
        TextColumn(" • "),
        TimeElapsedColumn(),
        transient=False,
    ) as progress:
        task = progress.add_task("running", start=True)
        start = time.perf_counter()
        rc = None

        with open(os.devnull, "wb") as devnull:
            proc = None
            try:
                proc = subprocess.Popen(cmd, stdout=devnull, stderr=devnull)
                rc = proc.wait()
            except KeyboardInterrupt:
                if proc and proc.poll() is None:
                    try:
                        proc.terminate()
                        proc.wait(timeout=5)
                    except Exception:
                        proc.kill()
                        proc.wait()
                rc = -130
            finally:
                if rc == 0:
                    progress.update(task, description=f"[green]✅ {label} Finished[/green]", completed=True)
                else:
                    code = rc if rc is not None else -1
                    progress.update(task, description=f"[red]❌ {label} Failed (code {code})[/red]", completed=True)
        elapsed = time.perf_counter() - start
    if rc == 0:
        print(f"✅ Finished in {elapsed:.1f}s")
    else:
        print(f"❌ Failed (code {rc}) after {elapsed:.1f}s")
    return rc, elapsed
