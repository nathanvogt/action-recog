/**
 * Utility type that transforms an interface into an async version
 * where every property is wrapped in a Promise.
 *
 * @example
 * interface User {
 *   id: string;
 *   name: string;
 *   age: number;
 * }
 *
 * type AsyncUser = AsyncInterface<User>;
 * // Result:
 * // {
 * //   id: Promise<string>;
 * //   name: Promise<string>;
 * //   age: Promise<number>;
 * // }
 */
export type AsyncInterface<T> = {
  [K in keyof T]: Promise<T[K]>;
};

/**
 * Utility type that transforms an interface with methods into an async version
 * where every method returns a Promise of its original return type.
 *
 * @example
 * interface UserService {
 *   name: string;
 *   getUser(id: string): User;
 *   updateUser(user: User): boolean;
 * }
 *
 * type AsyncUserService = AsyncMethods<UserService>;
 * // Result:
 * // {
 * //   name: Promise<string>;
 * //   getUser(id: string): Promise<User>;
 * //   updateUser(user: User): Promise<boolean>;
 * // }
 */
export type AsyncMethods<T> = {
  [K in keyof T]: T[K] extends (...args: any[]) => any
    ? (...args: Parameters<T[K]>) => Promise<ReturnType<T[K]>>
    : Promise<T[K]>;
};

/**
 * Alternative utility type that preserves optional properties
 * while wrapping each property type in a Promise.
 *
 * @example
 * interface User {
 *   id: string;
 *   name?: string;
 *   age: number;
 * }
 *
 * type AsyncUser = AsyncInterfacePreserveOptional<User>;
 * // Result:
 * // {
 * //   id: Promise<string>;
 * //   name?: Promise<string>;
 * //   age: Promise<number>;
 * // }
 */
export type AsyncInterfacePreserveOptional<T> = {
  [K in keyof T]: Promise<T[K]>;
};

/**
 * Utility type that makes all properties required and wraps them in Promises
 */
export type AsyncInterfaceRequired<T> = {
  [K in keyof T]-?: Promise<T[K]>;
};

/**
 * Utility type that makes all properties optional and wraps them in Promises
 */
export type AsyncInterfaceOptional<T> = {
  [K in keyof T]?: Promise<T[K]>;
};
