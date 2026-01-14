## Module architecture
```bash
core/
├── __init__.py          # Central imports (unchanged API)
├── descriptors/
│   ├── __init__.py      # Exports descriptor classes
│   ├── depth_buffer_descriptor.py
│   └── lfd_descriptor.py
├── mesh/
│   ├── __init__.py      # Exports mesh operations
│   ├── mesh_loader.py
│   ├── mesh_normalizer.py
│   ├── renderer.py
│   └── view_generator.py
└── similarity/
    ├── __init__.py      # Exports similarity engine
    └── similarity_engine.py
```