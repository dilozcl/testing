import os

class DirectoryCreator:
    def __init__(self, root_dir):
        self.root_dir = root_dir

    def create_directories(self, directories):
        for directory in directories:
            dir_path = os.path.join(self.root_dir, directory)
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
                print(f"Directory '{directory}' created.")
            else:
                print(f"Directory '{directory}' already exists.")

# Example usage:
root_directory = '/path/to/root'
directories_to_create = ['dir1', 'dir2', 'dir3']

dir_creator = DirectoryCreator(root_directory)
dir_creator.create_directories(directories_to_create)
