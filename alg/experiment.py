import sys
from pathlib import Path
import shutil
from alg.ga import GeneticAlgorithm
from alg.functions import functions
import shutil


def folder_deep_copy(source_folder, destination_folder):
    source_path = Path(source_folder)
    destination_path = Path(destination_folder)

    overwritten_files = []
    created_files = []
    non_found_files = []

    for source_file in source_path.glob("**/*"):
        if source_file.is_file():
            destination_file = destination_path / source_file.relative_to(source_path)

            if destination_file.exists():
                overwritten_files.append(destination_file)
            else:
                created_files.append(destination_file)

            destination_file.parent.mkdir(exist_ok=True, parents=True)
            shutil.copy2(source_file, destination_file)

    for destination_file in destination_path.glob("**/*"):
        if destination_file.is_file():
            source_file = source_path / destination_file.relative_to(destination_path)

            if not source_file.exists():
                non_found_files.append(destination_file)

    return overwritten_files, created_files, non_found_files


class Experiment:
    def __init__(self, target_function, n_dims, n_points, iter_factor, use_wandb,
                 n_ga_generations, ga_convergence_ratio, experiments_root):
        self.n_dims = n_dims
        self.n_points = n_points
        self.iter_factor = iter_factor
        self.target_function = target_function
        self.use_wandb = use_wandb
        self.experiments_root = experiments_root

        self.n_ga_generations = n_ga_generations
        self.ga_convergence_ratio = ga_convergence_ratio

        self.experiment_name = f'{self.target_function.name}_d={self.n_dims}_if={self.iter_factor}_p={self.n_points}'
        self.experiment_folder = Path(experiments_root) / self.experiment_name
        self.experiment_folder.mkdir(parents=True, exist_ok=True)

        self.pop_size = 32
        self.mutation_rate = 0.01
        self.cx_pool_size = 20

    def run(self):
        ga = GeneticAlgorithm(target_function=self.target_function, n_dims=self.n_dims,
                              pop_size=self.pop_size, mutation_rate=self.mutation_rate,
                              cx_pool_size=self.cx_pool_size, n_points=self.n_points,
                              iter_factor=self.iter_factor, n_bins=1000, tolerance=1e-3,
                              use_wandb=self.use_wandb)
        ga.run(self.n_ga_generations, self.ga_convergence_ratio, self.experiment_folder)


if __name__ == '__main__':
    # set per machine
    function_index = 0
    target_function = functions[function_index]  # 0 to 8

    # 4 multiple processes per machine
    # [2, 3, 4, 5]
    n_dims = 2  # 2 to 5

    experiments_root = Path('checkpoints')
    iter_factor = 500  # [50, 100, 250, 500]

    for n_points in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]:
        experiment = Experiment(target_function=target_function, n_dims=n_dims, n_points=n_points,
                                iter_factor=iter_factor, use_wandb=False,
                                n_ga_generations=100, ga_convergence_ratio=0.5,
                                experiments_root=experiments_root)
        experiment.run()

    if 'win' not in sys.platform:
        destination = Path('~/Nextcloud').expanduser() / f'aiwork' / f'checkpoints_{function_index}.zip'

        shutil.make_archive(str(destination), 'zip', experiments_root)
        # folder_deep_copy(source_folder=experiments_root, destination_folder=destination)
