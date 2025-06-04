# TrainDataset Server

This project provides a local HTTP server that exposes the `TrainDatasetLocal` functionality over HTTP, allowing browser-based applications to access local training data files through the `TrainDatasetRemote` class.

## Overview

The system consists of two main components:

1. **TrainDatasetServer** - A local HTTP server that provides REST API endpoints
2. **TrainDatasetRemote** - A client class that can be used in browsers to access the server

## Quick Start

### 1. Install Dependencies

First, install the required dependencies:

```bash
npm install
```

This will install Express.js, CORS, and their TypeScript types.

### 2. Start the Server

```bash
npm run server
```

This starts the server on `http://localhost:3001` by default. You can customize the port by setting the `PORT` environment variable:

```bash
PORT=3002 npm run server
```

You can also customize the dataset root directory:

```bash
DATASET_ROOT=/path/to/your/data npm run server
```

### 3. Test the Server

In another terminal, test the server functionality:

```bash
npm run test-remote-dataset
```

This will run a comprehensive test of all the remote dataset methods.

## API Endpoints

The server exposes the following REST API endpoints:

| Method | Endpoint                               | Description                            |
| ------ | -------------------------------------- | -------------------------------------- |
| GET    | `/api/root`                            | Get dataset root path                  |
| GET    | `/api/subjects`                        | List all subjects                      |
| GET    | `/api/exercises/:subject`              | List exercises for a specific subject  |
| GET    | `/api/exercises`                       | List all exercises                     |
| GET    | `/api/instances/:exercise`             | List instances for a specific exercise |
| GET    | `/api/rep-annotations/:subject`        | Get rep annotations for a subject      |
| GET    | `/api/instance/:subject/:exercise`     | Load instance data                     |
| GET    | `/api/poses/:subject/:exercise`        | Get pose array                         |
| GET    | `/api/has-exercise/:subject/:exercise` | Check if subject has exercise          |
| GET    | `/api/rep-timings/:subject/:exercise`  | Get rep timings                        |
| GET    | `/api/rep-segments/:subject/:exercise` | Get rep segments                       |
| GET    | `/health`                              | Health check endpoint                  |

## Using TrainDatasetRemote

The `TrainDatasetRemote` class provides an async interface that mirrors the `TrainDatasetLocal` class:

```typescript
import { TrainDatasetRemote } from "./libs/trainDataset/trainDataset.js";

// Create remote dataset instance
const remote = new TrainDatasetRemote("http://localhost:3001");

// All methods return promises
const subjects = await remote.listSubjects();
const exercises = await remote.listAllExercises();
const instance = await remote.loadInstance("subject1", "exercise1");
```

### In Browser Applications

You can use the `TrainDatasetRemote` class directly in React or other browser applications:

```typescript
import { TrainDatasetRemote } from "./libs/trainDataset/trainDataset";

function MyComponent() {
  const [subjects, setSubjects] = useState<string[]>([]);
  const remote = new TrainDatasetRemote("http://localhost:3001");

  useEffect(() => {
    remote.listSubjects().then(setSubjects);
  }, []);

  return (
    <div>
      <h2>Subjects:</h2>
      <ul>
        {subjects.map((subject) => (
          <li key={subject}>{subject}</li>
        ))}
      </ul>
    </div>
  );
}
```

## Configuration

### Environment Variables

- `PORT` - Server port (default: 3001)
- `DATASET_ROOT` - Path to training data directory (default: 'train')

### CORS

The server is configured with CORS enabled to allow browser requests from any origin. This is suitable for development but should be configured more restrictively for production use.

## Error Handling

All API endpoints include proper error handling:

- **404** - When requested data doesn't exist
- **500** - For server errors with detailed error messages

The `TrainDatasetRemote` class will throw errors with descriptive messages when API calls fail.

## Development

### File Structure

```
src/
├── server/
│   └── trainDatasetServer.ts    # Express server implementation
├── libs/
│   ├── trainDataset/
│   │   ├── trainDataset.ts      # Local and Remote classes
│   │   └── trainDatasetTypes.ts # Type definitions
│   └── utilTypes.ts             # Utility types including AsyncMethods
└── scripts/
    ├── startServer.ts           # Server startup script
    └── testRemoteDataset.ts     # Test script for remote functionality
```

### Type Safety

The `TrainDatasetRemote` class implements the `AsyncMethods<TrainDataset>` interface, which automatically converts all methods from the original `TrainDataset` interface to return promises while preserving type safety.

### Testing

Run the test suite to verify everything works:

```bash
# Start server in one terminal
npm run server

# Test in another terminal
npm run test-remote-dataset
```

## Troubleshooting

### Common Issues

1. **Server won't start**: Check if port 3001 is already in use
2. **CORS errors**: Make sure the server is running and accessible
3. **File not found errors**: Verify the `DATASET_ROOT` path is correct
4. **TypeScript errors**: Run `npm install` to ensure all dependencies are installed

### Debugging

The server logs all requests and errors to the console. Check the server output for detailed error messages.

## Production Considerations

For production use, consider:

- Configuring CORS more restrictively
- Adding authentication/authorization
- Adding rate limiting
- Using a process manager like PM2
- Adding HTTPS support
- Adding request validation and sanitization
